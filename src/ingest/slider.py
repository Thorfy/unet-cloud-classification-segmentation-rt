"""CIRA / RAMMB SLIDER GeoColor tiles (GOES-19, GOES-18, Meteosat 0deg)."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from typing import Any

import requests
from PIL import Image

from src.paths import USER_AGENT

SLIDER_HOSTS = (
    "https://slider.cira.colostate.edu",
    "https://rammb-slider.cira.colostate.edu",
)

SATELLITES = {
    "goes-19": "goes-19",
    "goes-18": "goes-18",
    "meteosat-disk": "meteosat-0deg",
}


def _headers() -> dict[str, str]:
    return {"User-Agent": USER_AGENT}


def latest_timestamp(satellite: str, host: str) -> str:
    url = f"{host}/data/json/{satellite}/full_disk/geocolor/latest_times.json"
    resp = requests.get(url, headers=_headers(), timeout=30)
    resp.raise_for_status()
    stamps = resp.json()["timestamps_int"]
    return str(stamps[0])


def _tile_url(host: str, sat: str, ts: str, zoom: str, row: int, col: int) -> str:
    year, month, day = ts[:4], ts[4:6], ts[6:8]
    return (
        f"{host}/data/imagery/{year}/{month}/{day}/"
        f"{sat}---full_disk/geocolor/{ts}/{zoom}/{row:03d}_{col:03d}.png"
    )


def _download_tile(url: str) -> Image.Image | None:
    try:
        resp = requests.get(url, headers=_headers(), timeout=30)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def fetch_geocolor(source: str = "goes-19", zoom: int = 1) -> dict[str, Any]:
    sat = SATELLITES[source]
    n = 2 ** zoom
    zoom_key = f"{zoom:02d}"
    last_error = None
    for host in SLIDER_HOSTS:
        try:
            ts = latest_timestamp(sat, host)
        except Exception as exc:
            last_error = exc
            continue
        tiles: list[tuple[int, int, Image.Image]] = []
        tile_size = None
        ok = True
        for row in range(n):
            for col in range(n):
                url = _tile_url(host, sat, ts, zoom_key, row, col)
                im = _download_tile(url)
                if im is None:
                    ok = False
                    break
                tile_size = im.size
                tiles.append((row, col, im))
            if not ok:
                break
        if not ok or not tiles or tile_size is None:
            continue
        Image.MAX_IMAGE_PIXELS = None
        tw, th = tile_size
        canvas = Image.new("RGB", (tw * n, th * n))
        for row, col, im in tiles:
            canvas.paste(im, (col * tw, row * th))
        captured = datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        return {
            "image": canvas,
            "source": source,
            "provider": f"CIRA SLIDER GeoColor ({sat})",
            "captured_at": captured.isoformat(),
            "url": _tile_url(host, sat, ts, zoom_key, 0, 0),
            "domain_gap": (
                "GeoColor CIRA, pas le CMI ABI truecolor CONUS utilise a l'entrainement des types."
            ),
        }
    raise RuntimeError(f"Impossible de recuperer SLIDER pour {source}: {last_error}")


def fetch_europe_geocolor() -> dict[str, Any]:
    """GeoColor Meteosat 0deg recadre sur l'Europe (disque plein, nord)."""
    payload = fetch_geocolor("meteosat-disk", zoom=2)
    image: Image.Image = payload["image"]
    w, h = image.size
    crop = image.crop((int(0.30 * w), int(0.02 * h), int(0.70 * w), int(0.42 * h)))
    payload["image"] = crop
    payload["source"] = "meteosat-europe"
    payload["provider"] = "CIRA SLIDER GeoColor (Meteosat 0deg, crop Europe)"
    payload["domain_gap"] = "GeoColor Meteosat recadre Europe (fraicheur SLIDER parfois de quelques heures)."
    return payload
