"""Dispatch live satellite sources."""

from __future__ import annotations

from typing import Any, Callable

from src.ingest.goes_aws import latest_conus_rgb
from src.ingest.metnorway import fetch_europe_auto, fetch_europe_infrared, fetch_europe_visible
from src.ingest.slider import fetch_europe_geocolor, fetch_geocolor
from src.ingest.star import fetch_goes19_star

SOURCES: dict[str, str] = {
    "europe": "Meteosat Europe auto (MET Norway visible / IR nuit)",
    "europe-visible": "Meteosat Europe visible (MET Norway)",
    "europe-ir": "Meteosat Europe infrared (MET Norway)",
    "meteosat-europe": "Meteosat Europe GeoColor (CIRA SLIDER)",
    "meteosat-disk": "Meteosat 0deg GeoColor disque (CIRA SLIDER)",
    "goes-19-cmi": "GOES-19 ABI truecolor CONUS (NOAA)",
    "goes-19": "GOES-19 GeoColor (CIRA SLIDER)",
    "goes-19-star": "GOES-19 GeoColor full disk (NOAA STAR)",
}


def fetch_latest(source: str = "europe") -> dict[str, Any]:
    fetchers: dict[str, Callable[[], dict[str, Any]]] = {
        "europe": fetch_europe_auto,
        "europe-visible": fetch_europe_visible,
        "europe-ir": fetch_europe_infrared,
        "meteosat-europe": fetch_europe_geocolor,
        "meteosat-disk": lambda: fetch_geocolor("meteosat-disk", zoom=1),
        "goes-19-cmi": latest_conus_rgb,
        "goes-19": lambda: fetch_geocolor("goes-19", zoom=1),
        "goes-19-star": fetch_goes19_star,
    }
    if source not in fetchers:
        raise KeyError(f"Source inconnue: {source}. Choisir parmi {list(SOURCES)}")
    return fetchers[source]()
