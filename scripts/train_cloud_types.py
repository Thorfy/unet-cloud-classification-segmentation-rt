#!/usr/bin/env python3
"""Entrainement U-Net types de nuages (GOES ACTP/COD)."""

from __future__ import annotations

import argparse
import json
import sys
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
from src.paths import GOES_SEG_DIR, RUNS_DIR, ensure_dirs
from src.unet import UNet


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


class TileDataset(Dataset):
    def __init__(self, split: str, augment: bool = False) -> None:
        self.img_dir = GOES_SEG_DIR / split / "images"
        self.mask_dir = GOES_SEG_DIR / split / "masks"
        self.names = sorted(p.stem for p in self.img_dir.glob("*.png"))
        self.augment = augment
        if not self.names:
            raise FileNotFoundError(f"Pas de tuiles dans {self.img_dir}")

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        rgb = np.asarray(Image.open(self.img_dir / f"{name}.png").convert("RGB"), dtype=np.float32) / 255.0
        mask = np.asarray(Image.open(self.mask_dir / f"{name}.png"), dtype=np.int64)
        if self.augment:
            if np.random.rand() < 0.5:
                rgb = np.ascontiguousarray(rgb[:, ::-1])
                mask = np.ascontiguousarray(mask[:, ::-1])
            k = int(np.random.randint(0, 4))
            if k:
                rgb = np.ascontiguousarray(np.rot90(rgb, k))
                mask = np.ascontiguousarray(np.rot90(mask, k))
        image = torch.from_numpy(rgb.transpose(2, 0, 1).copy())
        target = torch.from_numpy(mask.copy())
        return image, target


def class_weights(split: str = "train") -> torch.Tensor:
    hist = np.zeros(NUM_CLASSES, dtype=np.float64)
    mask_dir = GOES_SEG_DIR / split / "masks"
    for path in mask_dir.glob("*.png"):
        arr = np.asarray(Image.open(path))
        for i in range(NUM_CLASSES):
            hist[i] += int((arr == i).sum())
    hist = np.maximum(hist, 1.0)
    inv = 1.0 / np.log(1.2 + hist / hist.sum())
    inv = inv / inv.mean()
    return torch.tensor(inv, dtype=torch.float32)


@torch.no_grad()
def evaluate(model: UNet, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    inter = np.zeros(NUM_CLASSES, dtype=np.int64)
    union = np.zeros(NUM_CLASSES, dtype=np.int64)
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        pred = model(images).argmax(1)
        valid = targets != IGNORE
        correct += int(((pred == targets) & valid).sum().item())
        total += int(valid.sum().item())
        p = pred.cpu().numpy()
        t = targets.cpu().numpy()
        for i in range(NUM_CLASSES):
            pi = p == i
            ti = t == i
            inter[i] += int(np.logical_and(pi, ti).sum())
            union[i] += int(np.logical_or(pi, ti).sum())
    ious = [float(inter[i] / union[i]) if union[i] else 0.0 for i in range(NUM_CLASSES)]
    return {
        "acc": float(correct / max(total, 1)),
        "miou": float(np.mean(ious)),
        **{f"iou_{i}": ious[i] for i in range(NUM_CLASSES)},
    }


def main() -> None:
    _configure_stdio()
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = TileDataset("train", augment=True)
    valid_ds = TileDataset("valid", augment=False)
    drop_last = len(train_ds) >= args.batch_size
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    weights = class_weights("train").to(device)
    model = UNet(num_classes=NUM_CLASSES).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out_dir = RUNS_DIR / "unet-cloud-types"
    out_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for images, targets in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, targets, weight=weights, ignore_index=IGNORE)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item())
            n += 1
        sched.step()
        metrics = evaluate(model, valid_loader, device)
        metrics["loss"] = running / max(n, 1)
        history.append({"epoch": epoch, **metrics})
        print(
            f"epoch {epoch}: loss={metrics['loss']:.4f} acc={metrics['acc']:.3f} miou={metrics['miou']:.3f}",
            flush=True,
        )
        torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, out_dir / "last.pt")
        if metrics["miou"] > best:
            best = metrics["miou"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, out_dir / "best.pt")
            print(f"  best miou {best:.3f}", flush=True)

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Entrainement termine: {out_dir} (best miou={best:.3f})")


if __name__ == "__main__":
    main()
