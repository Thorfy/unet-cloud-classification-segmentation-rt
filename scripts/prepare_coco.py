#!/usr/bin/env python3
"""Convert 38/95-Cloud semantic masks to RF-DETR COCO instance format."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import COCO_DIR, RAW_DIR, ensure_dirs
from src.scene import classify_fraction

CATEGORY = {"id": 1, "name": "cloud", "supercategory": "atmosphere"}
MIN_AREA = 32


def stretch_to_uint8(arr: np.ndarray) -> np.ndarray:
    data = arr.astype(np.float32)
    valid = data > 0
    if not np.any(valid):
        return np.zeros(data.shape, dtype=np.uint8)
    lo, hi = np.percentile(data[valid], (2, 98))
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((data - lo) / (hi - lo), 0, 1)
    return (scaled * 255).astype(np.uint8)


def read_band(path: Path) -> np.ndarray:
    try:
        import tifffile

        return np.asarray(tifffile.imread(path))
    except Exception:
        im = Image.open(path)
        return np.asarray(im)


def stack_rgb(red: Path, green: Path, blue: Path) -> np.ndarray:
    r = stretch_to_uint8(read_band(red))
    g = stretch_to_uint8(read_band(green))
    b = stretch_to_uint8(read_band(blue))
    if r.ndim == 3:
        r = r[..., 0]
    if g.ndim == 3:
        g = g[..., 0]
    if b.ndim == 3:
        b = b[..., 0]
    h = min(r.shape[0], g.shape[0], b.shape[0])
    w = min(r.shape[1], g.shape[1], b.shape[1])
    return np.stack([r[:h, :w], g[:h, :w], b[:h, :w]], axis=-1)


def mask_to_instances(mask: np.ndarray, min_area: int = MIN_AREA) -> list[dict]:
    binary = (mask > 0).astype(np.uint8)
    num, labels = cv2.connectedComponents(binary)
    instances = []
    for idx in range(1, num):
        component = (labels == idx).astype(np.uint8)
        area = int(component.sum())
        if area < min_area:
            continue
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segmentation = []
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            flat = cnt.reshape(-1, 2).astype(float).flatten().tolist()
            if len(flat) >= 6:
                segmentation.append(flat)
        if not segmentation:
            ys, xs = np.where(component)
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            segmentation = [[float(x0), float(y0), float(x1), float(y0), float(x1), float(y1), float(x0), float(y1)]]
        ys, xs = np.where(component)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        w, h = x1 - x0 + 1, y1 - y0 + 1
        instances.append(
            {
                "bbox": [float(x0), float(y0), float(w), float(h)],
                "area": float(area),
                "segmentation": segmentation,
            }
        )
    return instances


def coco_skeleton() -> dict:
    return {
        "info": {"description": "38/95-Cloud converted for RF-DETR-Seg", "version": "1.0"},
        "licenses": [],
        "categories": [CATEGORY],
        "images": [],
        "annotations": [],
    }


def add_sample(coco: dict, image_id: int, file_name: str, rgb: np.ndarray, mask: np.ndarray, ann_id_start: int) -> int:
    h, w = rgb.shape[:2]
    coco["images"].append({"id": image_id, "file_name": file_name, "width": w, "height": h})
    ann_id = ann_id_start
    for inst in mask_to_instances(mask):
        coco["annotations"].append(
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": inst["bbox"],
                "area": inst["area"],
                "iscrowd": 0,
                "segmentation": inst["segmentation"],
            }
        )
        ann_id += 1
    return ann_id


def save_split(coco: dict, split_dir: Path) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")


def collect_band_patches(train_root: Path) -> list[dict]:
    red_dir = None
    for cand in train_root.rglob("train_red"):
        if cand.is_dir():
            red_dir = cand
            break
    if red_dir is None:
        return []
    parent = red_dir.parent
    items = []
    for red in sorted(red_dir.glob("*")):
        if not red.is_file():
            continue
        name = red.name
        suffix = name.split("red", 1)[-1] if "red" in name.lower() else name
        def sibling(folder: str, prefix: str) -> Path | None:
            d = parent / folder
            if not d.is_dir():
                return None
            exact = d / (prefix + suffix)
            if exact.exists():
                return exact
            stem = red.stem.replace("red_", "").replace("train_red_", "")
            hits = list(d.glob(f"*{stem}*"))
            return hits[0] if hits else None

        green = sibling("train_green", "green") or sibling("train_green", "Green")
        blue = sibling("train_blue", "blue") or sibling("train_blue", "Blue")
        gt = sibling("train_gt", "gt") or sibling("train_gt", "GT")
        if green is None or blue is None or gt is None:
            continue
        items.append({"red": red, "green": green, "blue": blue, "gt": gt, "stem": red.stem})
    return items


def is_informative(rgb: np.ndarray, max_black: float = 0.80) -> bool:
    black = np.all(rgb == 0, axis=-1).mean()
    return float(black) <= max_black


def export_from_bands(train_roots: list[Path], max_train: int | None, seed: int, valid_ratio: float) -> dict:
    samples = []
    for root in train_roots:
        samples.extend(collect_band_patches(root))
    rng = random.Random(seed)
    rng.shuffle(samples)
    if max_train:
        samples = samples[:max_train]
    n_valid = max(1, int(len(samples) * valid_ratio)) if samples else 0
    valid_set = set(id(s) for s in samples[:n_valid])
    stats = {"train": 0, "valid": 0, "skipped": 0, "classes": {}}
    coco = {"train": coco_skeleton(), "valid": coco_skeleton()}
    ann_ids = {"train": 1, "valid": 1}
    img_ids = {"train": 1, "valid": 1}
    for sample in tqdm(samples, desc="COCO bands"):
        rgb = stack_rgb(sample["red"], sample["green"], sample["blue"])
        if not is_informative(rgb):
            stats["skipped"] += 1
            continue
        mask = read_band(sample["gt"])
        if mask.ndim == 3:
            mask = mask[..., 0]
        split = "valid" if id(sample) in valid_set else "train"
        file_name = f"{sample['stem']}.jpg"
        out = COCO_DIR / split / file_name
        Image.fromarray(rgb).save(out, quality=95)
        ann_ids[split] = add_sample(coco[split], img_ids[split], file_name, rgb, mask, ann_ids[split])
        img_ids[split] += 1
        stats[split] += 1
        label = classify_fraction(float((mask > 0).mean()))
        stats["classes"][label] = stats["classes"].get(label, 0) + 1
    for split in ("train", "valid"):
        save_split(coco[split], COCO_DIR / split)
    return stats


def export_from_hf(hf_dir: Path, max_train: int | None, seed: int, valid_ratio: float) -> dict:
    images = sorted((hf_dir / "images").glob("*"))
    rng = random.Random(seed)
    rng.shuffle(images)
    if max_train:
        images = images[:max_train]
    n_valid = max(1, int(len(images) * valid_ratio)) if images else 0
    valid_names = {p.name for p in images[:n_valid]}
    stats = {"train": 0, "valid": 0, "skipped": 0, "classes": {}}
    coco = {"train": coco_skeleton(), "valid": coco_skeleton()}
    ann_ids = {"train": 1, "valid": 1}
    img_ids = {"train": 1, "valid": 1}
    for img_path in tqdm(images, desc="COCO HF 38-Cloud"):
        mask_path = hf_dir / "masks" / img_path.name
        if not mask_path.exists():
            stats["skipped"] += 1
            continue
        rgb = np.asarray(Image.open(img_path).convert("RGB"))
        mask = np.asarray(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        if not is_informative(rgb):
            stats["skipped"] += 1
            continue
        split = "valid" if img_path.name in valid_names else "train"
        file_name = img_path.stem + ".jpg"
        Image.fromarray(rgb).save(COCO_DIR / split / file_name, quality=95)
        ann_ids[split] = add_sample(coco[split], img_ids[split], file_name, rgb, mask, ann_ids[split])
        img_ids[split] += 1
        stats[split] += 1
        label = classify_fraction(float((mask > 0).mean()))
        stats["classes"][label] = stats["classes"].get(label, 0) + 1
    for split in ("train", "valid"):
        save_split(coco[split], COCO_DIR / split)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=3000, help="Sous-ensemble nonempty (0 = tout)")
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    ensure_dirs()
    import shutil

    for split in ("train", "valid"):
        split_dir = COCO_DIR / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

    max_train = None if args.max_train == 0 else args.max_train
    combined = RAW_DIR / "95cloud_combined_training"
    k38 = RAW_DIR / "kaggle_38"
    k95 = RAW_DIR / "kaggle_95"
    hf95 = RAW_DIR / "95cloud_hf"
    hf38 = RAW_DIR / "38cloud_hf"

    band_roots = [p for p in (combined, k38, k95) if p.exists() and any(p.rglob("train_red"))]
    if band_roots:
        stats = export_from_bands(band_roots, max_train, args.seed, args.valid_ratio)
        source = "38/95-Cloud Kaggle bands"
    elif (hf95 / "images").exists():
        stats = export_from_hf(hf95, max_train, args.seed, args.valid_ratio)
        source = "95-Cloud Hugging Face"
    elif (hf38 / "images").exists():
        stats = export_from_hf(hf38, max_train, args.seed, args.valid_ratio)
        source = "38-Cloud Hugging Face"
    else:
        print("Aucune donnee brute. Lancez: python scripts/download_datasets.py")
        return 1

    (COCO_DIR / "PREPARE_STATUS.json").write_text(
        json.dumps({"source": source, **stats}, indent=2), encoding="utf-8"
    )
    print(json.dumps({"source": source, **stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
