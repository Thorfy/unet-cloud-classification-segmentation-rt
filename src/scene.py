"""Scene-level cloud classification derived from a binary mask."""

from typing import Tuple

CLEAR_MAX = 0.10
PARTLY_MAX = 0.50

LABELS = ("clear", "partly_cloudy", "cloudy")


def cloud_fraction(mask) -> float:
    import numpy as np

    arr = np.asarray(mask)
    if arr.size == 0:
        return 0.0
    binary = arr > 0
    return float(binary.mean())


def classify_fraction(fraction: float) -> str:
    if fraction < CLEAR_MAX:
        return "clear"
    if fraction < PARTLY_MAX:
        return "partly_cloudy"
    return "cloudy"


def classify_mask(mask) -> Tuple[str, float]:
    frac = cloud_fraction(mask)
    return classify_fraction(frac), frac
