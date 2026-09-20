"""NOAA STAR GeoColor full-disk JPEG (GOES-East fallback)."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from typing import Any

import requests
from PIL import Image

from src.paths import USER_AGENT

LATEST = "https://cdn.star.nesdis.noaa.gov/GOES19/ABI/FD/GEOCOLOR/1808x1808.jpg"
MAX_SIDE = 1808


def _downscale(image: Image.Image, max_side: int = MAX_SIDE) -> Image.Image:
    w, h = image.size
    longest = max(w, h)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    return image.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)


def fetch_goes19_star() -> dict[str, Any]:
    Image.MAX_IMAGE_PIXELS = None
    resp = requests.get(LATEST, headers={"User-Agent": USER_AGENT}, timeout=60)
    resp.raise_for_status()
    image = _downscale(Image.open(BytesIO(resp.content)).convert("RGB"))
    captured = resp.headers.get("Last-Modified")
    return {
        "image": image,
        "source": "goes-19-star",
        "provider": "NOAA STAR GOES-19 GeoColor",
        "captured_at": captured or datetime.now(timezone.utc).isoformat(),
        "url": LATEST,
        "domain_gap": (
            "GeoColor full disk STAR, pas le CMI ABI truecolor CONUS d'entrainement."
        ),
    }
