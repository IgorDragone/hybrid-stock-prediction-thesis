import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd  # noqa: E402

from src.config import DATASETS_DIR, ROOT_DIR, load_universe  # noqa: E402
from src.config.io import load_yaml  # noqa: E402

RUN_SPEC_PATH = Path(__file__).with_name("run_spec.yaml")
RUN_SPEC = load_yaml(RUN_SPEC_PATH)

GENERAL_CFG = RUN_SPEC["general"]
DB_BUILD_CFG = RUN_SPEC["db_build"]
PREPROCESSING_CFG = RUN_SPEC.get("preprocessing", {})
MODELING_CFG = RUN_SPEC.get("modeling", {})
BACKTEST_CFG = RUN_SPEC.get("backtest", {})

MASTER_ID = GENERAL_CFG["dataset_id"]
OUT_DIR = DATASETS_DIR / MASTER_ID

UNIVERSE_PATH = ROOT_DIR / DB_BUILD_CFG.get("universe_path", "configs/universe.yaml")
MASTER_TICKERS: list[str] = []
for entry in load_universe(UNIVERSE_PATH):
    MASTER_TICKERS.extend(entry.get("tickers", []))

START_DATE = pd.to_datetime(DB_BUILD_CFG["start_date"])
END_DATE = pd.to_datetime(DB_BUILD_CFG["end_date"])
START_DATE_STR = START_DATE.strftime("%Y-%m-%d")
END_DATE_STR = END_DATE.strftime("%Y-%m-%d")

WALK_FORWARD_CFG = {
    "train_years": MODELING_CFG["train_years"],
    "test_months": MODELING_CFG["test_months"],
    "embargo_months": MODELING_CFG["embargo_months"],
}

TARGET_COL = MODELING_CFG.get("target_col", "target_3m")

