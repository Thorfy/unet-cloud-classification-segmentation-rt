"""MET Norway geostationary images (Meteosat Europe)."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from typing import Any

import numpy as np
import requests
from PIL import Image

from src.paths import USER_AGENT

BASE = "https://api.met.no/weatherapi/geosatellite/1.4/"
DARK_MEAN = 40.0


def _headers(accept: str = "image/png") -> dict[str, str]:
    return {"User-Agent": USER_AGENT, "Accept": accept}


def _get_image(params: dict[str, str]) -> tuple[Image.Image, requests.Response]:
    resp = requests.get(BASE, params=params, headers=_headers(), timeout=60)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB"), resp


def _is_dark(image: Image.Image) -> bool:
    arr = np.asarray(image)
    return float(arr.mean()) < DARK_MEAN


def list_available(area: str = "europe", img_type: str = "visible", size: str = "normal") -> list[dict[str, Any]]:
    resp = requests.get(
        BASE + "available.json",
        params={"area": area, "type": img_type, "size": size},
        headers=_headers("application/json"),
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, list) else []


def fetch_europe_at(time_iso: str, img_type: str = "visible") -> dict[str, Any]:
    image, resp = _get_image(
        {"area": "europe", "type": img_type, "size": "normal", "time": time_iso}
    )
    return {
        "image": image,
        "source": "europe" if img_type == "visible" else "europe-ir",
        "provider": f"MET Norway Geosatellite 1.4 (Meteosat {img_type})",
        "captured_at": time_iso,
        "url": resp.url,
        "domain_gap": "Meteosat Europe (MET Norway), pas GOES-19 CONUS.",
        "channel": img_type,
    }


def fetch_europe_visible() -> dict[str, Any]:
    image, resp = _get_image({"area": "europe", "type": "visible", "size": "normal"})
    captured = resp.headers.get("Last-Modified") or datetime.now(timezone.utc).isoformat()
    return {
        "image": image,
        "source": "europe-visible",
        "provider": "MET Norway Geosatellite 1.4 (Meteosat visible)",
        "captured_at": captured,
        "url": resp.url,
        "domain_gap": "Meteosat Europe visible ~1 km (MET Norway).",
        "channel": "visible",
    }


def fetch_europe_infrared() -> dict[str, Any]:
    image, resp = _get_image({"area": "europe", "type": "infrared", "size": "normal"})
    captured = resp.headers.get("Last-Modified") or datetime.now(timezone.utc).isoformat()
    return {
        "image": image,
        "source": "europe-ir",
        "provider": "MET Norway Geosatellite 1.4 (Meteosat infrared)",
        "captured_at": captured,
        "url": resp.url,
        "domain_gap": "Infrared Meteosat de nuit: les types restent moins fiables qu'en visible diurne.",
        "channel": "infrared",
    }


def fetch_europe_auto() -> dict[str, Any]:
    """Visible le jour, infrared si l'Europe est trop sombre."""
    image, resp = _get_image({"area": "europe", "type": "visible", "size": "normal"})
    captured = resp.headers.get("Last-Modified") or datetime.now(timezone.utc).isoformat()
    if _is_dark(image):
        night = fetch_europe_infrared()
        night["source"] = "europe"
        night["domain_gap"] = (
            "Nuit en Europe: visible Meteosat trop sombre, bascule automatique en infrared."
        )
        return night
    return {
        "image": image,
        "source": "europe",
        "provider": "MET Norway Geosatellite 1.4 (Meteosat visible)",
        "captured_at": captured,
        "url": resp.url,
        "domain_gap": "Meteosat Europe visible ~1 km (MET Norway).",
        "channel": "visible",
    }
