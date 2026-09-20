"""GOES-19 ABI L2 public NOAA sur AWS (MCMIP RGB, ACTP phase, COD)."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

from src.paths import GOES_RAW_DIR, USER_AGENT, ensure_dirs

BUCKET = "https://noaa-goes19.s3.amazonaws.com"
SAT = "noaa-goes19"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
START_RE = re.compile(r"_s(\d{11})")

PRODUCTS = {
    "MCMIPC": "ABI-L2-MCMIPC",
    "ACTPC": "ABI-L2-ACTPC",
    "CODC": "ABI-L2-CODC",
    "ACHAC": "ABI-L2-ACHAC",
}


def _headers() -> dict[str, str]:
    return {"User-Agent": USER_AGENT}


def parse_start(key: str) -> str | None:
    match = START_RE.search(key)
    return match.group(1) if match else None


def list_keys(prefix: str, limit: int = 400) -> list[str]:
    keys: list[str] = []
    token: str | None = None
    while len(keys) < limit:
        params: dict[str, str] = {
            "list-type": "2",
            "prefix": prefix,
            "max-keys": "200",
        }
        if token:
            params["continuation-token"] = token
        resp = requests.get(f"{BUCKET}/", params=params, headers=_headers(), timeout=60)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        for node in root.findall("s3:Contents/s3:Key", NS):
            if node.text:
                keys.append(node.text)
                if len(keys) >= limit:
                    break
        truncated = root.find("s3:IsTruncated", NS)
        if truncated is None or (truncated.text or "").lower() != "true":
            break
        cont = root.find("s3:NextContinuationToken", NS)
        token = cont.text if cont is not None else None
        if not token:
            break
    return keys


def download_key(key: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    url = f"{BUCKET}/{key}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, headers=_headers(), stream=True, timeout=180) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    tmp.replace(dest)
    return dest


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def candidate_slots(days: int = 8, hours: tuple[int, ...] = (15, 17, 19, 21)) -> list[tuple[int, int, int]]:
    """Liste (year, doy, hour) UTC recents, diurnes CONUS."""
    now = _utc_now()
    slots: list[tuple[int, int, int]] = []
    for back in range(days):
        day = now - timedelta(days=back)
        doy = int(day.strftime("%j"))
        year = day.year
        for hour in hours:
            if back == 0 and hour > now.hour:
                continue
            slots.append((year, doy, hour))
    return slots


def pick_scene_keys(year: int, doy: int, hour: int) -> dict[str, str] | None:
    found: dict[str, str] = {}
    index: dict[str, dict[str, str]] = {}
    for name, product in PRODUCTS.items():
        prefix = f"{product}/{year}/{doy:03d}/{hour:02d}/"
        for key in list_keys(prefix, limit=80):
            stamp = parse_start(key)
            if stamp is None:
                continue
            index.setdefault(stamp, {})[name] = key
    for stamp, parts in sorted(index.items()):
        if "MCMIPC" in parts and "ACTPC" in parts:
            found = parts
            break
    return found or None


def download_scenes(max_scenes: int = 24, days: int = 8) -> list[Path]:
    ensure_dirs()
    scenes: list[Path] = []
    for year, doy, hour in candidate_slots(days=days):
        if len(scenes) >= max_scenes:
            break
        parts = pick_scene_keys(year, doy, hour)
        if not parts:
            continue
        stamp = parse_start(parts["MCMIPC"])
        if stamp is None:
            continue
        dest_dir = GOES_RAW_DIR / "scenes" / stamp
        dest_dir.mkdir(parents=True, exist_ok=True)
        for name, key in parts.items():
            download_key(key, dest_dir / f"{name}.nc")
        scenes.append(dest_dir)
    return scenes


def _open_nc(path: Path):
    from netCDF4 import Dataset

    return Dataset(path, "r")


def _first_var(ds, names: tuple[str, ...], fill):
    for name in names:
        if name in ds.variables:
            data = ds.variables[name][:]
            if np.ma.isMaskedArray(data):
                return np.ma.filled(data, fill)
            return np.asarray(data)
    raise KeyError(f"Aucune variable parmi {names}: {list(ds.variables)}")


def _masked_to_float(arr: np.ndarray) -> np.ndarray:
    data = np.ma.filled(np.ma.array(arr), np.nan).astype(np.float32)
    return data


def cmi_to_rgb(c01: np.ndarray, c02: np.ndarray, c03: np.ndarray) -> np.ndarray:
    """Truecolor GOES: R=C02, B=C01, G synthetique, gamma 0.5."""
    r = np.clip(_masked_to_float(c02), 0, 1)
    b = np.clip(_masked_to_float(c01), 0, 1)
    veg = np.clip(_masked_to_float(c03), 0, 1)
    g = np.clip(0.45 * r + 0.10 * veg + 0.45 * b, 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.power(np.clip(rgb, 0, 1), 0.5)
    return np.nan_to_num(rgb * 255.0, nan=0.0).astype(np.uint8)


def read_cmi_rgb(path: Path) -> np.ndarray:
    ds = _open_nc(path)
    try:
        c01 = _first_var(ds, ("CMI_C01", "C01"), fill=np.nan)
        c02 = _first_var(ds, ("CMI_C02", "C02"), fill=np.nan)
        c03 = _first_var(ds, ("CMI_C03", "C03"), fill=np.nan)
        return cmi_to_rgb(c01, c02, c03)
    finally:
        ds.close()


def read_phase(path: Path) -> np.ndarray:
    ds = _open_nc(path)
    try:
        return _first_var(ds, ("Phase", "ACTP"), fill=255)
    finally:
        ds.close()


def read_height(path: Path) -> np.ndarray:
    ds = _open_nc(path)
    try:
        raw = _first_var(ds, ("HT", "Height", "ACHA"), fill=np.nan)
        arr = np.asarray(raw, dtype=np.float32)
        arr[(arr < 0) | (arr > 25000)] = np.nan
        return arr
    finally:
        ds.close()


def download_acha_for_existing() -> list[Path]:
    """Complete ACHAC (hauteur) pour les scenes deja telechargees."""
    updated: list[Path] = []
    root = GOES_RAW_DIR / "scenes"
    if not root.is_dir():
        return updated
    for scene in sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit()):
        dest = scene / "ACHAC.nc"
        if dest.is_file() and dest.stat().st_size > 0:
            updated.append(scene)
            continue
        stamp = scene.name
        year, doy, hour = int(stamp[:4]), int(stamp[4:7]), int(stamp[7:9])
        prefix = f"ABI-L2-ACHAC/{year}/{doy:03d}/{hour:02d}/"
        match = None
        for key in list_keys(prefix, limit=80):
            if parse_start(key) == stamp:
                match = key
                break
        if match is None:
            continue
        download_key(match, dest)
        updated.append(scene)
    return updated


def read_cod(path: Path) -> np.ndarray:
    ds = _open_nc(path)
    try:
        raw = _first_var(ds, ("COD", "CODC"), fill=np.nan)
        arr = np.asarray(raw, dtype=np.float32)
        arr[(arr < 0) | (arr > 200)] = np.nan
        return arr
    finally:
        ds.close()


def resize_to(arr: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    if arr.shape[:2] == shape_hw:
        return arr
    import cv2

    interp = cv2.INTER_NEAREST if arr.dtype in (np.uint8, np.int16, np.int32) else cv2.INTER_LINEAR
    return cv2.resize(arr, (shape_hw[1], shape_hw[0]), interpolation=interp)


def latest_conus_rgb() -> dict[str, Any]:
    """Dernier MCMIPC CONUS -> RGB truecolor (meme domaine que l'entrainement)."""
    now = _utc_now()
    last_error: Exception | None = None
    for back_h in range(0, 8):
        when = now - timedelta(hours=back_h)
        prefix = f"ABI-L2-MCMIPC/{when.year}/{int(when.strftime('%j')):03d}/{when.hour:02d}/"
        try:
            keys = [k for k in list_keys(prefix, limit=40) if k.endswith(".nc")]
        except Exception as exc:
            last_error = exc
            continue
        if not keys:
            continue
        key = keys[-1]
        dest = GOES_RAW_DIR / "live" / Path(key).name
        download_key(key, dest)
        rgb = read_cmi_rgb(dest)
        stamp = parse_start(key) or ""
        captured = now.isoformat()
        if len(stamp) >= 11:
            try:
                captured = datetime.strptime(stamp[:11], "%Y%j%H%M").replace(tzinfo=timezone.utc).isoformat()
            except ValueError:
                pass
        return {
            "image": Image.fromarray(rgb),
            "source": "goes-19-cmi",
            "provider": "NOAA GOES-19 ABI MCMIP CONUS",
            "captured_at": captured,
            "url": f"{BUCKET}/{key}",
            "domain_gap": (
                "Meme domaine que l'entrainement: ABI CMI truecolor CONUS ~2 km."
            ),
        }
    raise RuntimeError(f"Aucun MCMIPC recent: {last_error}")
