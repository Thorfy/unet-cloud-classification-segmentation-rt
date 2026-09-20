"""Inference segmentation de types de nuages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.cloud_types import (
    CLASSES,
    IGNORE,
    NUM_CLASSES,
    class_fractions,
    cloud_fraction,
    overlay_types,
)
from src.paths import RUNS_DIR
from src.instances import annotate_instances, mask_to_instances
from src.scene import classify_fraction
from src.unet import UNet

DEFAULT_CHECKPOINT = RUNS_DIR / "unet-cloud-types" / "best.pt"
EUROPE_CHECKPOINT = RUNS_DIR / "unet-cloud-types-europe" / "best.pt"
TILE = 384
STRIDE = 320


def _best_checkpoint() -> Path | None:
    candidates = [
        EUROPE_CHECKPOINT,
        RUNS_DIR / "unet-cloud-types-europe" / "last.pt",
        DEFAULT_CHECKPOINT,
        RUNS_DIR / "unet-cloud-types" / "last.pt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _head_classes(state: dict) -> int | None:
    weight = state.get("head.weight")
    if weight is None:
        return None
    return int(weight.shape[0])


def load_type_model(checkpoint: Path | None = None, device: str | None = None):
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tried = [checkpoint] if checkpoint is not None else [
        EUROPE_CHECKPOINT,
        RUNS_DIR / "unet-cloud-types-europe" / "last.pt",
        DEFAULT_CHECKPOINT,
        RUNS_DIR / "unet-cloud-types" / "last.pt",
    ]
    last_error = "Aucun checkpoint unet-cloud-types (best.pt)."
    for ckpt in tried:
        if ckpt is None or not Path(ckpt).is_file():
            continue
        payload = torch.load(ckpt, map_location=dev, weights_only=False)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        n = _head_classes(state)
        if n is not None and n != NUM_CLASSES:
            last_error = f"{ckpt} a {n} classes, le modele actuel en attend {NUM_CLASSES}."
            continue
        model = UNet(num_classes=NUM_CLASSES)
        model.load_state_dict(state)
        model.to(dev)
        model.eval()
        return model, str(ckpt), dev
    raise FileNotFoundError(last_error)


def _pad32(arr: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = arr.shape[:2]
    ph = (32 - h % 32) % 32
    pw = (32 - w % 32) % 32
    if ph == 0 and pw == 0:
        return arr, (h, w)
    return np.pad(arr, ((0, ph), (0, pw), (0, 0)), mode="reflect"), (h, w)


@torch.no_grad()
def predict_mask(model: UNet, rgb: np.ndarray, device: torch.device) -> np.ndarray:
    """Argmax tuile par tuile, moyenne des logits sur les recouvrements."""
    image, (h, w) = _pad32(np.asarray(rgb))
    hh, ww = image.shape[:2]
    logits = np.zeros((NUM_CLASSES, hh, ww), dtype=np.float32)
    counts = np.zeros((hh, ww), dtype=np.float32)
    ys = list(range(0, max(1, hh - TILE + 1), STRIDE))
    xs = list(range(0, max(1, ww - TILE + 1), STRIDE))
    if hh <= TILE:
        ys = [0]
    elif ys[-1] != hh - TILE:
        ys.append(hh - TILE)
    if ww <= TILE:
        xs = [0]
    elif xs[-1] != ww - TILE:
        xs.append(ww - TILE)

    tensor_full = None
    if hh <= TILE and ww <= TILE:
        tensor_full = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0
        pred = model(tensor_full.unsqueeze(0).to(device))[0].cpu().numpy()
        return pred[:, :h, :w].argmax(0).astype(np.uint8)

    for y in ys:
        for x in xs:
            tile = image[y : y + TILE, x : x + TILE]
            if tile.shape[0] < TILE or tile.shape[1] < TILE:
                pad = np.pad(
                    tile,
                    ((0, TILE - tile.shape[0]), (0, TILE - tile.shape[1]), (0, 0)),
                    mode="reflect",
                )
            else:
                pad = tile
            inp = torch.from_numpy(pad.transpose(2, 0, 1).copy()).float() / 255.0
            pred = model(inp.unsqueeze(0).to(device))[0].cpu().numpy()
            th, tw = tile.shape[:2]
            logits[:, y : y + th, x : x + tw] += pred[:, :th, :tw]
            counts[y : y + th, x : x + tw] += 1.0
    counts = np.maximum(counts, 1e-6)
    mask = (logits / counts).argmax(0)
    return mask[:h, :w].astype(np.uint8)


def run_type_inference(
    image: Image.Image,
    model=None,
    device=None,
    max_side: int = 1536,
    min_instance_area: int | None = None,
) -> dict[str, Any]:
    origin = "provided"
    if model is None:
        model, origin, device = load_type_model()
    rgb = image.convert("RGB")
    w, h = rgb.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / float(longest)
        rgb = rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR)
    arr = np.asarray(rgb)
    mask = predict_mask(model, arr, device)
    space = arr.max(axis=-1) <= 8
    mask = mask.copy()
    mask[space] = IGNORE
    frac = cloud_fraction(mask)
    instances = mask_to_instances(mask, min_area=min_instance_area)
    return {
        "image": rgb,
        "mask": mask,
        "overlay": Image.fromarray(overlay_types(arr, mask)),
        "instances": instances,
        "instance_overlay": Image.fromarray(annotate_instances(arr, mask, instances)),
        "n_instances": len(instances),
        "cloud_fraction": frac,
        "scene_class": classify_fraction(frac),
        "fractions": class_fractions(mask),
        "weights": origin,
    }


def legend_rows() -> list[tuple[str, str]]:
    rows = []
    for idx, (_key, label, rgb) in CLASSES.items():
        if idx == 0:
            continue
        hexcol = "#{:02x}{:02x}{:02x}".format(*rgb)
        rows.append((hexcol, label))
    return rows
