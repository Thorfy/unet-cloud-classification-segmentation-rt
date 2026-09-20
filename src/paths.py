from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("CLOUD_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
COCO_DIR = DATA_DIR / "coco"
GOES_RAW_DIR = RAW_DIR / "goes19"
GOES_SEG_DIR = DATA_DIR / "goes_seg"
EUROPE_RAW_DIR = RAW_DIR / "meteosat_europe"
EUROPE_SEG_DIR = DATA_DIR / "europe_seg"
RUNS_DIR = ROOT / "runs"
WEIGHTS_DIR = ROOT / "weights"
KAGGLE_TOKEN = Path.home() / ".kaggle" / "kaggle.json"
USER_AGENT = "cloud-net-rfdetr/1.0 (educational cloud detection; local project)"


def ensure_dirs() -> Path:
    for d in (
        DATA_DIR,
        RAW_DIR,
        COCO_DIR,
        GOES_RAW_DIR,
        GOES_SEG_DIR,
        EUROPE_RAW_DIR,
        EUROPE_SEG_DIR,
        RUNS_DIR,
        WEIGHTS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
    return DATA_DIR
