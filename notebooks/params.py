import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config import DATASETS_DIR  # noqa: E402
from src.config.io import load_yaml  # noqa: E402

RUN_SPEC_PATH = Path(__file__).with_name("run_spec.yaml")
CFG = load_yaml(RUN_SPEC_PATH)

DATASET_ID = CFG["dataset_id"]
OUT_DIR = DATASETS_DIR / DATASET_ID

