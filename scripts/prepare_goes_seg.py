#!/usr/bin/env python3
"""Tuile les scenes GOES (RGB CMI + types ACTP/COD) pour le U-Net."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cloud_types import IGNORE, NUM_CLASSES, actp_to_types
from src.ingest.goes_aws import read_cmi_rgb, read_cod, read_height, read_phase, resize_to
from src.paths import GOES_RAW_DIR, GOES_SEG_DIR, ensure_dirs

TILE = 384
MIN_EARTH = 0.40


def _tiles(h: int, w: int, size: int) -> list[tuple[int, int]]:
    ys = list(range(0, max(1, h - size + 1), size))
    xs = list(range(0, max(1, w - size + 1), size))
    if ys[-1] != h - size and h >= size:
        ys.append(h - size)
    if xs[-1] != w - size and w >= size:
        xs.append(w - size)
    return [(y, x) for y in ys for x in xs]


def _earth_fraction(rgb: np.ndarray) -> float:
    return float((rgb.max(axis=-1) > 8).mean())


def process_scene(scene: Path, size: int) -> list[tuple[np.ndarray, np.ndarray, str]]:
    rgb_path = scene / "MCMIPC.nc"
    phase_path = scene / "ACTPC.nc"
    if not rgb_path.is_file() or not phase_path.is_file():
        return []
    rgb = read_cmi_rgb(rgb_path)
    phase = read_phase(phase_path)
    phase = resize_to(phase, rgb.shape[:2])
    cod = None
    if (scene / "CODC.nc").is_file():
        try:
            cod = resize_to(read_cod(scene / "CODC.nc"), rgb.shape[:2])
        except Exception:
            cod = None
    height = None
    if (scene / "ACHAC.nc").is_file():
        try:
            height = resize_to(read_height(scene / "ACHAC.nc"), rgb.shape[:2])
        except Exception:
            height = None
    types = actp_to_types(phase, cod, height)
    out: list[tuple[np.ndarray, np.ndarray, str]] = []
    if min(rgb.shape[:2]) < size:
        return out
    for y, x in _tiles(*rgb.shape[:2], size):
        tile_rgb = rgb[y : y + size, x : x + size]
        tile_m = types[y : y + size, x : x + size]
        if _earth_fraction(tile_rgb) < MIN_EARTH:
            continue
        if float((tile_m == IGNORE).mean()) > 0.85:
            continue
        name = f"{scene.name}_{y:04d}_{x:04d}"
        out.append((tile_rgb, tile_m, name))
    return out


def save_split(items: list[tuple[np.ndarray, np.ndarray, str]], split: str) -> None:
    img_dir = GOES_SEG_DIR / split / "images"
    mask_dir = GOES_SEG_DIR / split / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    for rgb, mask, name in items:
        Image.fromarray(rgb).save(img_dir / f"{name}.png")
        Image.fromarray(mask).save(mask_dir / f"{name}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile", type=int, default=TILE)
    parser.add_argument("--valid-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    ensure_dirs()
    scenes = sorted(
        p for p in (GOES_RAW_DIR / "scenes").glob("*") if p.is_dir() and p.name.isdigit()
    )
    if not scenes:
        raise SystemExit("Aucune scene dans data/raw/goes19/scenes")

    random.seed(args.seed)
    random.shuffle(scenes)
    n_valid = max(1, int(round(len(scenes) * args.valid_frac)))
    valid_scenes = set(scenes[:n_valid])

    for split in ("train", "valid"):
        img_dir = GOES_SEG_DIR / split / "images"
        mask_dir = GOES_SEG_DIR / split / "masks"
        if img_dir.exists():
            for f in img_dir.glob("*.png"):
                f.unlink()
        if mask_dir.exists():
            for f in mask_dir.glob("*.png"):
                f.unlink()

    counts = {"train": 0, "valid": 0}
    hist = np.zeros(NUM_CLASSES, dtype=np.int64)
    for scene in tqdm(scenes, desc="tuilage"):
        split = "valid" if scene in valid_scenes else "train"
        tiles = process_scene(scene, args.tile)
        save_split(tiles, split)
        counts[split] += len(tiles)
        for _, mask, _ in tiles:
            for i in range(NUM_CLASSES):
                hist[i] += int((mask == i).sum())

    status = {
        "scenes": len(scenes),
        "tiles": counts,
        "pixels": {str(i): int(hist[i]) for i in range(NUM_CLASSES)},
    }
    (GOES_SEG_DIR / "PREPARE_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
