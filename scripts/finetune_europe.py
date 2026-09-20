#!/usr/bin/env python3
"""Fine-tune le U-Net GOES sur Meteosat Europe (MET Norway, visible diurne)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cloud_types import IGNORE, NUM_CLASSES
from src.infer_types import load_type_model, predict_mask
from src.ingest.metnorway import fetch_europe_at, list_available
from src.paths import EUROPE_RAW_DIR, EUROPE_SEG_DIR, RUNS_DIR, ensure_dirs
from src.unet import UNet

TILE = 384
MIN_CONF = 0.55
DAY_HOURS = range(7, 16)


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _is_daytime(time_iso: str) -> bool:
    try:
        stamp = datetime.fromisoformat(time_iso.replace("Z", "+00:00"))
    except ValueError:
        return False
    return stamp.hour in DAY_HOURS


def download_daytime(max_images: int) -> list[Path]:
    ensure_dirs()
    items = [x for x in list_available("europe", "visible") if _is_daytime(x.get("params", {}).get("time", ""))]
    items = items[-max_images:]
    saved: list[Path] = []
    for item in tqdm(items, desc="download europe"):
        time_iso = item["params"]["time"]
        safe = time_iso.replace(":", "").replace("-", "")
        dest = EUROPE_RAW_DIR / f"{safe}.png"
        if not dest.is_file():
            payload = fetch_europe_at(time_iso, "visible")
            payload["image"].save(dest)
        saved.append(dest)
    return saved


def _tiles(h: int, w: int, size: int) -> list[tuple[int, int]]:
    ys = list(range(0, max(1, h - size + 1), size))
    xs = list(range(0, max(1, w - size + 1), size))
    if h >= size and ys[-1] != h - size:
        ys.append(h - size)
    if w >= size and xs[-1] != w - size:
        xs.append(w - size)
    return [(y, x) for y in ys for x in xs]


@torch.no_grad()
def label_images(paths: list[Path], teacher) -> None:
    model, _, device = teacher
    img_dir = EUROPE_SEG_DIR / "train" / "images"
    mask_dir = EUROPE_SEG_DIR / "train" / "masks"
    val_img = EUROPE_SEG_DIR / "valid" / "images"
    val_mask = EUROPE_SEG_DIR / "valid" / "masks"
    for folder in (img_dir, mask_dir, val_img, val_mask):
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.png"):
            old.unlink()

    n_train = 0
    n_valid = 0
    for idx, path in enumerate(tqdm(paths, desc="pseudo-labels")):
        rgb = np.asarray(Image.open(path).convert("RGB"))
        logits_ok = True
        try:
            mask = predict_mask(model, rgb, device)
        except Exception as exc:
            print("skip", path, exc)
            logits_ok = False
        if not logits_ok:
            continue
        # Confiance via un second passage softmax sur tuiles n'est pas dans predict_mask.
        # On ignore les pixels espace / quasi noirs.
        dark = rgb.max(axis=-1) <= 12
        mask = mask.copy()
        mask[dark] = IGNORE
        split_img = val_img if idx % 7 == 0 else img_dir
        split_mask = val_mask if idx % 7 == 0 else mask_dir
        if min(rgb.shape[:2]) < TILE:
            continue
        for y, x in _tiles(*rgb.shape[:2], TILE):
            tile_rgb = rgb[y : y + TILE, x : x + TILE]
            tile_m = mask[y : y + TILE, x : x + TILE]
            if float((tile_m == IGNORE).mean()) > 0.9:
                continue
            name = f"{path.stem}_{y:04d}_{x:04d}"
            Image.fromarray(tile_rgb).save(split_img / f"{name}.png")
            Image.fromarray(tile_m).save(split_mask / f"{name}.png")
            if split_img is val_img:
                n_valid += 1
            else:
                n_train += 1
    print(json.dumps({"train_tiles": n_train, "valid_tiles": n_valid}))


class TileDataset(Dataset):
    def __init__(self, split: str, augment: bool = False) -> None:
        self.img_dir = EUROPE_SEG_DIR / split / "images"
        self.mask_dir = EUROPE_SEG_DIR / split / "masks"
        self.names = sorted(p.stem for p in self.img_dir.glob("*.png"))
        self.augment = augment
        if not self.names:
            raise FileNotFoundError(f"Pas de tuiles Europe dans {self.img_dir}")

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        rgb = np.asarray(Image.open(self.img_dir / f"{name}.png").convert("RGB"), dtype=np.float32) / 255.0
        mask = np.asarray(Image.open(self.mask_dir / f"{name}.png"), dtype=np.int64)
        if self.augment and np.random.rand() < 0.5:
            rgb = np.ascontiguousarray(rgb[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
        return torch.from_numpy(rgb.transpose(2, 0, 1).copy()), torch.from_numpy(mask.copy())


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    inter = np.zeros(NUM_CLASSES, dtype=np.int64)
    union = np.zeros(NUM_CLASSES, dtype=np.int64)
    correct = total = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        pred = model(images).argmax(1)
        valid = targets != IGNORE
        correct += int(((pred == targets) & valid).sum().item())
        total += int(valid.sum().item())
        p, t = pred.cpu().numpy(), targets.cpu().numpy()
        for i in range(NUM_CLASSES):
            inter[i] += int(np.logical_and(p == i, t == i).sum())
            union[i] += int(np.logical_or(p == i, t == i).sum())
    ious = [float(inter[i] / union[i]) if union[i] else 0.0 for i in range(NUM_CLASSES)]
    return {"acc": float(correct / max(total, 1)), "miou": float(np.mean(ious))}


def train(epochs: int, batch_size: int, lr: float) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = load_type_model(checkpoint=RUNS_DIR / "unet-cloud-types" / "best.pt", device=str(device))
    model = UNet(num_classes=NUM_CLASSES).to(device)
    payload = torch.load(RUNS_DIR / "unet-cloud-types" / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(payload["model"] if isinstance(payload, dict) and "model" in payload else payload)

    train_ds = TileDataset("train", augment=True)
    valid_ds = TileDataset("valid", augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=len(train_ds) >= batch_size)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    out_dir = RUNS_DIR / "unet-cloud-types-europe"
    out_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for images, targets in tqdm(train_loader, desc=f"europe {epoch}/{epochs}"):
            images, targets = images.to(device), targets.to(device)
            loss = F.cross_entropy(model(images), targets, ignore_index=IGNORE)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item())
            n += 1
        sched.step()
        metrics = evaluate(model, valid_loader, device)
        metrics["loss"] = running / max(n, 1)
        print(f"epoch {epoch}: loss={metrics['loss']:.4f} acc={metrics['acc']:.3f} miou={metrics['miou']:.3f}", flush=True)
        torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, out_dir / "last.pt")
        if metrics["miou"] >= best:
            best = metrics["miou"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, out_dir / "best.pt")
            print(f"  best europe miou {best:.3f}", flush=True)
    print(f"Fine-tune Europe termine: {out_dir} (best miou={best:.3f})")
    _ = teacher


def main() -> None:
    _configure_stdio()
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    if args.skip_download:
        paths = sorted(EUROPE_RAW_DIR.glob("*.png"))
    else:
        paths = download_daytime(args.max_images)
    if not paths:
        raise SystemExit("Aucune image Europe diurne.")
    teacher = load_type_model(checkpoint=RUNS_DIR / "unet-cloud-types" / "best.pt")
    label_images(paths, teacher)
    train(args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
