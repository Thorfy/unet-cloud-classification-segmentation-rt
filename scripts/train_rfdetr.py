#!/usr/bin/env python3
"""Fine-tune RF-DETR-Seg Nano on 38/95-Cloud COCO."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import COCO_DIR, RUNS_DIR, ensure_dirs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", default=str(RUNS_DIR / "rfdetr-seg"))
    parser.add_argument(
        "--pretrain",
        default="",
        help="Checkpoint .pth a charger avant le fine-tune (ex. 38-Cloud).",
    )
    args = parser.parse_args()
    ensure_dirs()

    train_ann = COCO_DIR / "train" / "_annotations.coco.json"
    valid_ann = COCO_DIR / "valid" / "_annotations.coco.json"
    if not train_ann.is_file() or not valid_ann.is_file():
        print("Dataset COCO introuvable. Lancez:")
        print("  python scripts/download_datasets.py")
        print("  python scripts/prepare_coco.py")
        return 1

    import torch
    from rfdetr import RFDETRSegNano

    batch_size = args.batch_size
    if isinstance(batch_size, str) and batch_size.isdigit():
        batch_size = int(batch_size)
    if batch_size == "auto" and not torch.cuda.is_available():
        batch_size = 2
        print("CUDA indisponible: batch_size=2 (auto requiert CUDA).")

    print(f"cuda={torch.cuda.is_available()} batch={batch_size}")
    print(f"Train RFDETRSegNano on {COCO_DIR} -> {args.output}")
    pretrain = args.pretrain.strip()
    if pretrain:
        print(f"Init depuis {pretrain}")
        model = RFDETRSegNano(pretrain_weights=pretrain, num_classes=1)
    else:
        model = RFDETRSegNano()
    train_kwargs = dict(
        dataset_dir=str(COCO_DIR),
        epochs=args.epochs,
        batch_size=batch_size,
        lr=args.lr,
        output_dir=args.output,
    )
    if not torch.cuda.is_available():
        train_kwargs["grad_accum_steps"] = 4
    model.train(**train_kwargs)
    print("Entrainement termine:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
