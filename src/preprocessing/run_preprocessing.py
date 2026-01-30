import pandas as pd
from src.preprocessing.preprocess_panel import (
    preprocess_panel, PreprocessConfig
)
from src.data.config import PROCESSED_DIR

def main():
    file_path = PROCESSED_DIR / "financial_database.parquet"
    df = pd.read_parquet(file_path)

    cfg = PreprocessConfig(
        macro_cols=["CPIAUCSL", "FEDFUNDS", "GDP"],
        fundamental_cols=[...],
        growth_cols=[...],
        margin_cols=[...],
        min_non_na_ratio=0.90,
    )

    df_p = preprocess_panel(df, config=cfg)

    df_p.to_parquet(
        PROCESSED_DIR / "panel_preprocessed.parquet",
        index=False
    )

if __name__ == "__main__":
    main()
