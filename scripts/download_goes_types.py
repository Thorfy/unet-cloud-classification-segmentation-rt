#!/usr/bin/env python3
"""Telecharge des paires GOES-19 MCMIP + ACTP (+ COD) depuis AWS NOAA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingest.goes_aws import download_scenes
from src.paths import ensure_dirs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-scenes", type=int, default=24)
    parser.add_argument("--days", type=int, default=8)
    args = parser.parse_args()
    ensure_dirs()
    scenes = download_scenes(max_scenes=args.max_scenes, days=args.days)
    print(f"Scenes GOES telechargees: {len(scenes)}")
    for path in scenes:
        print(f"  {path}")
    if not scenes:
        raise SystemExit("Aucune scene trouvee sur AWS.")


if __name__ == "__main__":
    main()
