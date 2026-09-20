#!/usr/bin/env python3
"""Download 38-Cloud and 95-Cloud.

Primary: Kaggle (requires %USERPROFILE%\\.kaggle\\kaggle.json).
Fallback: Hugging Face jaygala223/38-cloud-dataset for the 38-Cloud train split.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import KAGGLE_TOKEN, RAW_DIR, ensure_dirs

KAGGLE_38 = "sorour/38cloud-cloud-segmentation-in-satellite-images"
KAGGLE_95 = "sorour/95cloud-cloud-segmentation-on-satellite-images"
HF_38 = "jaygala223/38-cloud-dataset"
HF_95 = "jaygala223/95-cloud-train-only-v1"

TRAIN_SUBDIRS = (
    "train_red",
    "train_green",
    "train_blue",
    "train_nir",
    "train_gt",
)


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def kaggle_available() -> bool:
    if not KAGGLE_TOKEN.is_file():
        return False
    return shutil.which("kaggle") is not None or True


def download_kaggle(slug: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            "-m",
            "kaggle",
            "datasets",
            "download",
            "-d",
            slug,
            "-p",
            str(dest),
            "--force",
        ]
    )
    zips = sorted(dest.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zips:
        raise FileNotFoundError(f"Aucun zip Kaggle pour {slug} dans {dest}")
    zip_path = zips[0]
    extract_dir = dest / zip_path.stem
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extraction {zip_path} -> {extract_dir}", flush=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    return extract_dir


def merge_training_dirs(sources: list[Path], combined: Path) -> None:
    combined.mkdir(parents=True, exist_ok=True)
    for sub in TRAIN_SUBDIRS:
        target = combined / sub
        target.mkdir(parents=True, exist_ok=True)
        for src in sources:
            folder = _find_named(src, sub)
            if folder is None:
                continue
            for item in folder.iterdir():
                dest = target / item.name
                if dest.exists():
                    continue
                if item.is_file():
                    shutil.copy2(item, dest)
    print(f"Train fusionne dans {combined}")


def _find_named(root: Path, name: str) -> Path | None:
    hits = [p for p in root.rglob(name) if p.is_dir()]
    return hits[0] if hits else None


def download_hf(repo: str, dest: Path, max_samples: int | None = None) -> Path:
    from datasets import load_dataset
    from tqdm import tqdm

    out_img = dest / "images"
    out_mask = dest / "masks"
    out_img.mkdir(parents=True, exist_ok=True)
    out_mask.mkdir(parents=True, exist_ok=True)
    print(f"Telechargement Hugging Face {repo} ...", flush=True)
    ds = load_dataset(repo, split="train")
    n = len(ds) if max_samples is None else min(max_samples, len(ds))
    for i in tqdm(range(n), desc=repo.split("/")[-1]):
        row = ds[i]
        image = row["image"]
        label = row["label"]
        image.save(out_img / f"patch_{i:05d}.png")
        label.save(out_mask / f"patch_{i:05d}.png")
    (dest / "SOURCE.txt").write_text(
        f"{repo}\nrows={n}\nnote=image/mask RGB pour RF-DETR\n",
        encoding="utf-8",
    )
    print(f"HF {repo}: {n} paires dans {dest}")
    return dest


def download_hf_38(dest: Path, max_samples: int | None = None) -> Path:
    return download_hf(HF_38, dest, max_samples)


def write_status(lines: list[str]) -> None:
    path = RAW_DIR / "DOWNLOAD_STATUS.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Download 38-Cloud and 95-Cloud")
    parser.add_argument("--hf-only", action="store_true", help="Skip Kaggle, use HF 38-Cloud")
    parser.add_argument("--max-hf", type=int, default=None)
    args = parser.parse_args()
    ensure_dirs()
    status: list[str] = []

    kaggle_ok = KAGGLE_TOKEN.is_file() and not args.hf_only
    if not kaggle_ok:
        msg = (
            "Kaggle API absente: placez kaggle.json dans "
            f"{KAGGLE_TOKEN} (compte Kaggle > Settings > Create token)."
        )
        print(msg, flush=True)
        status.append(msg)
        status.append("Repli Hugging Face: 95-Cloud train (prioritaire) + 38-Cloud si besoin.")
        hf95 = RAW_DIR / "95cloud_hf"
        download_hf(HF_95, hf95, max_samples=args.max_hf)
        status.append(f"OK HF 95-Cloud -> {hf95}")
        hf38 = RAW_DIR / "38cloud_hf"
        if not (hf38 / "images").exists():
            download_hf(HF_38, hf38, max_samples=args.max_hf)
            status.append(f"OK HF 38-Cloud -> {hf38}")
        write_status(status)
        return 0

    try:
        d38 = download_kaggle(KAGGLE_38, RAW_DIR / "kaggle_38")
        d95 = download_kaggle(KAGGLE_95, RAW_DIR / "kaggle_95")
        combined = RAW_DIR / "95cloud_combined_training"
        sources = [p for p in (d38, d95) if p.exists()]
        merge_training_dirs(sources, combined)
        status.append(f"OK Kaggle 38-Cloud -> {d38}")
        status.append(f"OK Kaggle 95-Cloud -> {d95}")
        status.append(f"OK fusion train -> {combined}")
        write_status(status)
        return 0
    except Exception as exc:
        status.append(f"Echec Kaggle: {exc}")
        print(status[-1], flush=True)
        print("Repli Hugging Face 95-Cloud...", flush=True)
        hf95 = RAW_DIR / "95cloud_hf"
        download_hf(HF_95, hf95, max_samples=args.max_hf)
        status.append(f"OK HF 95-Cloud -> {hf95}")
        write_status(status)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
