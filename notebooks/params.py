import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd  # noqa: E402

from src.config import DATASETS_DIR, ROOT_DIR, load_universe  # noqa: E402
from src.config.io import load_yaml  # noqa: E402

RUN_SPEC_PATH = Path(__file__).with_name("run_spec.yaml")
CFG = load_yaml(RUN_SPEC_PATH)

MASTER_ID = CFG["dataset_id"]
OUT_DIR = DATASETS_DIR / MASTER_ID

UNIVERSE_PATH = ROOT_DIR / CFG.get("universe_path", "configs/universe.yaml")
MASTER_TICKERS: list[str] = []
for entry in load_universe(UNIVERSE_PATH):
    MASTER_TICKERS.extend(entry.get("tickers", []))

START_DATE = pd.to_datetime(CFG["start_date"])
END_DATE = pd.to_datetime(CFG["end_date"])
START_DATE_STR = START_DATE.strftime("%Y-%m-%d")
END_DATE_STR = END_DATE.strftime("%Y-%m-%d")
