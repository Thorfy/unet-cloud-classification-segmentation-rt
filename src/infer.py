"""RF-DETR-Seg inference + scene classification from cloud masks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.paths import RUNS_DIR
from src.scene import classify_fraction

DEFAULT_CHECKPOINT = RUNS_DIR / "rfdetr-seg" / "checkpoint_best_total.pth"


def _best_checkpoint() -> Path | None:
    candidates = [
        RUNS_DIR / "rfdetr-seg-95" / "checkpoint_best_total.pth",
        RUNS_DIR / "rfdetr-seg-95" / "checkpoint_best_ema.pth",
        DEFAULT_CHECKPOINT,
        RUNS_DIR / "rfdetr-seg" / "checkpoint_best_ema.pth",
        RUNS_DIR / "rfdetr-seg" / "checkpoint_best_regular.pth",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_model(checkpoint: Path | None = None):
    from rfdetr import RFDETRSegNano

    ckpt = checkpoint or _best_checkpoint()
    if ckpt is not None:
        return RFDETRSegNano(pretrain_weights=str(ckpt), num_classes=1), str(ckpt)
    return RFDETRSegNano(), "coco-pretrained"


def _union_mask(detections, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if detections is None:
        return mask
    masks = getattr(detections, "mask", None)
    if masks is None:
        return mask
    for instance in masks:
        arr = np.asarray(instance)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.shape[:2] != (height, width):
            continue
        mask[arr.astype(bool)] = 255
    return mask


MAX_INFER_SIDE = 1280


def _fit(image: Image.Image, max_side: int = MAX_INFER_SIDE) -> Image.Image:
    w, h = image.size
    longest = max(w, h)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    return image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR)


def run_inference(image: Image.Image, model=None, threshold: float = 0.5) -> dict[str, Any]:
    if model is None:
        model, origin = load_model()
    else:
        origin = "provided"
    rgb = _fit(image.convert("RGB"))
    np_img = np.asarray(rgb).copy()
    detections = model.predict(np_img, threshold=threshold)
    h, w = np_img.shape[:2]
    mask = _union_mask(detections, h, w)
    fraction = float((mask > 0).mean()) if mask.size else 0.0
    label = classify_fraction(fraction)
    n_instances = 0
    if detections is not None and getattr(detections, "xyxy", None) is not None:
        n_instances = int(len(detections.xyxy))
    return {
        "image": rgb,
        "mask": mask,
        "detections": detections,
        "cloud_fraction": fraction,
        "scene_class": label,
        "n_instances": n_instances,
        "weights": origin,
    }


def annotate(result: dict[str, Any]) -> Image.Image:
    import supervision as sv

    base = np.asarray(result["image"])
    detections = result["detections"]
    if detections is None:
        return result["image"]
    annotated = sv.MaskAnnotator(opacity=0.45).annotate(base.copy(), detections)
    annotated = sv.BoxAnnotator(thickness=2).annotate(annotated, detections)
    return Image.fromarray(annotated)
