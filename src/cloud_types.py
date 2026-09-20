"""Taxonomie nuages enrichie: ACTP (phase) + COD (epaisseur) + ACHA (hauteur).

Inspiree du Cloud Type NWC SAF (bas / moyen / haut / cirrus), derivable
des produits GOES-19 publics.
"""

from __future__ import annotations

import numpy as np

IGNORE = 255

# id -> (cle, libelle FR, RGB overlay)
CLASSES: dict[int, tuple[str, str, tuple[int, int, int]]] = {
    0: ("clear", "Clair", (40, 40, 40)),
    1: ("fog", "Brouillard / tres bas", (220, 200, 70)),
    2: ("low", "Nuages bas (St/Sc/Cu)", (40, 110, 230)),
    3: ("mid", "Nuages moyens (Ac/As)", (70, 210, 230)),
    4: ("mixed", "Phase mixte", (70, 200, 90)),
    5: ("high_opaque", "Hauts opaques (Cs/Ns)", (200, 50, 60)),
    6: ("very_high", "Tres hauts / convectif (Cb)", (160, 40, 180)),
    7: ("cirrus_thin", "Cirrus tres mince", (250, 210, 230)),
    8: ("cirrus", "Cirrus", (250, 150, 210)),
    9: ("cirrus_thick", "Cirrus epais", (220, 70, 160)),
    10: ("unknown", "Inconnu", (120, 120, 120)),
}

NUM_CLASSES = len(CLASSES)
UNKNOWN_ID = 10
CLOUD_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9)

ACTP_CLEAR = 0
ACTP_WATER = 1
ACTP_SUPERCOOLED = 2
ACTP_MIXED = 3
ACTP_ICE = 4
ACTP_UNKNOWN = 5

H_FOG = 1000.0
H_LOW = 2500.0
H_MID = 6000.0
H_CB = 10000.0
COD_VERY_THIN = 1.0
COD_CIRRUS = 2.0
COD_THICK_CIRRUS = 4.0


def id_to_name(class_id: int) -> str:
    return CLASSES.get(int(class_id), CLASSES[UNKNOWN_ID])[1]


def palette() -> np.ndarray:
    table = np.zeros((256, 3), dtype=np.uint8)
    for idx, (_, _, rgb) in CLASSES.items():
        table[idx] = rgb
    return table


def actp_to_types(
    phase: np.ndarray,
    cod: np.ndarray | None = None,
    height: np.ndarray | None = None,
) -> np.ndarray:
    """Phase + COD + hauteur (m) -> ids CLASSES. Fill -> 255."""
    phase = np.asarray(phase)
    out = np.full(phase.shape, IGNORE, dtype=np.uint8)
    valid = np.isfinite(phase.astype(np.float32))
    p = np.zeros(phase.shape, dtype=np.int16)
    p[valid] = phase[valid].astype(np.int16)

    h = None
    if height is not None:
        h = np.asarray(height, dtype=np.float32)
        if h.shape != phase.shape:
            raise ValueError(f"HT shape {h.shape} != Phase {phase.shape}")

    c = None
    if cod is not None:
        c = np.asarray(cod, dtype=np.float32)
        if c.shape != phase.shape:
            raise ValueError(f"COD shape {c.shape} != Phase {phase.shape}")

    out[valid & (p == ACTP_CLEAR)] = 0
    out[valid & (p == ACTP_UNKNOWN)] = UNKNOWN_ID
    out[valid & (p == ACTP_MIXED)] = 4

    liquid = valid & ((p == ACTP_WATER) | (p == ACTP_SUPERCOOLED))
    if h is not None:
        fog = liquid & np.isfinite(h) & (h < H_FOG) & (h >= 0)
        low = liquid & np.isfinite(h) & (h >= H_FOG) & (h < H_LOW)
        mid = liquid & np.isfinite(h) & (h >= H_LOW)
        no_h = liquid & ~np.isfinite(h)
        out[fog] = 1
        out[low] = 2
        out[mid] = 3
        out[no_h & (p == ACTP_WATER)] = 2
        out[no_h & (p == ACTP_SUPERCOOLED)] = 3
    else:
        out[valid & (p == ACTP_WATER)] = 2
        out[valid & (p == ACTP_SUPERCOOLED)] = 3

    ice = valid & (p == ACTP_ICE)
    if c is None and h is None:
        out[ice] = 5
        return out

    if c is not None:
        very_thin = ice & np.isfinite(c) & (c >= 0) & (c <= COD_VERY_THIN)
        thin = ice & np.isfinite(c) & (c > COD_VERY_THIN) & (c <= COD_CIRRUS)
        thick_ci = ice & np.isfinite(c) & (c > COD_CIRRUS) & (c <= COD_THICK_CIRRUS)
        opaque = ice & (~very_thin) & (~thin) & (~thick_ci)
        out[very_thin] = 7
        out[thin] = 8
        out[thick_ci] = 9
        if h is not None:
            cb = opaque & np.isfinite(h) & (h >= H_CB)
            high = opaque & ~cb
            out[cb] = 6
            out[high] = 5
        else:
            out[opaque] = 5
    else:
        if h is not None:
            out[ice & np.isfinite(h) & (h >= H_CB)] = 6
            out[ice & ~(np.isfinite(h) & (h >= H_CB))] = 5
        else:
            out[ice] = 5
    return out


def class_fractions(mask: np.ndarray) -> dict[str, float]:
    arr = np.asarray(mask)
    known = arr != IGNORE
    n = int(known.sum())
    if n == 0:
        return {CLASSES[i][0]: 0.0 for i in range(NUM_CLASSES)}
    return {CLASSES[i][0]: float((arr == i).sum()) / n for i in range(NUM_CLASSES)}


def cloud_fraction(mask: np.ndarray) -> float:
    fr = class_fractions(mask)
    return float(sum(fr[CLASSES[i][0]] for i in CLOUD_IDS))


def overlay_types(rgb: np.ndarray, mask: np.ndarray, opacity: float = 0.45) -> np.ndarray:
    base = np.asarray(rgb)
    if base.ndim == 2:
        base = np.stack([base] * 3, axis=-1)
    out = base[..., :3].astype(np.float32).copy()
    colors = palette()
    m = np.asarray(mask)
    if m.shape[:2] != out.shape[:2]:
        raise ValueError("mask/image size mismatch")
    paintable = (m != IGNORE) & (m != 0)
    tint = colors[np.where(paintable, m, 0)].astype(np.float32)
    out[paintable] = (1.0 - opacity) * out[paintable] + opacity * tint[paintable]
    return np.clip(out, 0, 255).astype(np.uint8)
