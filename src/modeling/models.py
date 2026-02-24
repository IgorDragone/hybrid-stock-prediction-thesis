from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ModelIO:
    date_col: str = "date"
    ticker_col: str = "ticker"
    target_col: str = "target_3m"         # continuous target
    ret_col_for_ic: str = "fwd_ret_3m"  # only for IC diagnostics


def build_ridge_pipeline(alpha: float = 1.0, random_state: int = 42) -> Pipeline:
    """Ridge regression with scaling (leakage-safe via Pipeline)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=alpha, random_state=random_state))
    ])


def build_tree_model(
    random_state: int = 42,
    max_depth: int = 3,
    learning_rate: float = 0.05,
    max_iter: int = 300,
) -> HistGradientBoostingRegressor:
    """Histogram Gradient Boosting (fast, strong baseline, no scaling needed)."""
    return HistGradientBoostingRegressor(
        loss="squared_error",
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=random_state,
    )


def fit_predict_oos_scores(
    splits: List[Tuple[pd.DataFrame, pd.DataFrame, dict]],
    features: List[str],
    model,
    io: ModelIO,
    score_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit on each train fold, predict on each test fold, return concatenated OOS scores."""
    out_parts = []
    fold_rows = []

    for fold_id, (train_df, test_df, info) in enumerate(splits):
        missing_train = [c for c in features + [io.target_col] if c not in train_df.columns]
        missing_test = [c for c in features if c not in test_df.columns]
        if missing_train:
            raise ValueError(f"Missing in train fold {fold_id}: {missing_train}")
        if missing_test:
            raise ValueError(f"Missing in test fold {fold_id}: {missing_test}")

        X_train = train_df[features]
        y_train = train_df[io.target_col]
        X_test = test_df[features]

        model.fit(X_train, y_train)
        score = model.predict(X_test)

        tmp = test_df.copy()
        tmp[score_col] = score

        out_parts.append(tmp)
        fold_rows.append({"fold": fold_id, **info, "n_train": len(train_df), "n_test": len(test_df)})

    oos_df = pd.concat(out_parts, axis=0).sort_values([io.date_col, io.ticker_col]).reset_index(drop=True)
    folds_df = pd.DataFrame(fold_rows)
    return oos_df, folds_df
