from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ModelIO:
    date_col: str = "date"
    ticker_col: str = "ticker"
    target_col: str = "target"            # original {-1,0,+1}
    y_col: str = "y"                      # mapped {0,1,2}
    ret_col_for_ic: str = "fwd_ret_3_6m"  # only for IC diagnostics


def map_target_to_classes(df: pd.DataFrame, io: ModelIO) -> pd.DataFrame:
    out = df.copy()
    mapping = {-1: 0, 0: 1, 1: 2}
    if io.target_col not in out.columns:
        raise ValueError(f"Missing target column: {io.target_col}")
    out[io.y_col] = out[io.target_col].map(mapping)
    if out[io.y_col].isna().any():
        bad = out.loc[out[io.y_col].isna(), io.target_col].unique()
        raise ValueError(f"Found unmapped target values: {bad}")
    out[io.y_col] = out[io.y_col].astype("int8")
    return out


def build_logistic_pipeline(random_state: int = 42) -> Pipeline:
    """Multinomial logistic regression with scaling (leakage-safe via Pipeline)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            C=1.0,
            random_state=random_state,
        ))
    ])



def build_tree_model(
    random_state: int = 42,
    max_depth: int = 3,
    learning_rate: float = 0.05,
    max_iter: int = 300,
) -> HistGradientBoostingClassifier:
    """Histogram Gradient Boosting (fast, strong baseline, no scaling needed)."""
    return HistGradientBoostingClassifier(
        loss="log_loss",
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=random_state,
    )


def proba_to_expected_score(proba: np.ndarray) -> np.ndarray:
    """Convert class probabilities to a continuous ranking score in [-1, 1]."""
    if proba.ndim != 2 or proba.shape[1] != 3:
        raise ValueError("Expected proba with shape (n_samples, 3).")
    return proba[:, 2] * 1.0 + proba[:, 1] * 0.0 + proba[:, 0] * (-1.0)


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
        missing_train = [c for c in features + [io.y_col] if c not in train_df.columns]
        missing_test = [c for c in features if c not in test_df.columns]
        if missing_train:
            raise ValueError(f"Missing in train fold {fold_id}: {missing_train}")
        if missing_test:
            raise ValueError(f"Missing in test fold {fold_id}: {missing_test}")

        X_train = train_df[features]
        y_train = train_df[io.y_col]
        X_test = test_df[features]

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        score = proba_to_expected_score(proba)

        tmp = test_df.copy()
        tmp[score_col] = score

        out_parts.append(tmp)
        fold_rows.append({"fold": fold_id, **info, "n_train": len(train_df), "n_test": len(test_df)})

    oos_df = pd.concat(out_parts, axis=0).sort_values([io.date_col, io.ticker_col]).reset_index(drop=True)
    folds_df = pd.DataFrame(fold_rows)
    return oos_df, folds_df
