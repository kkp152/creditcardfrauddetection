from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fraud_rag.config import DATA_DIR, FEATURE_COLUMNS, TARGET_COLUMN


def load_creditcard_csv(path: Path | None = None) -> pd.DataFrame:
    path = path or (DATA_DIR / "creditcard.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download creditcard.csv from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place it under data/."
        )
    df = pd.read_csv(path)
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df


def load_and_split(
    path: Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler, pd.DataFrame]:
    df = load_creditcard_csv(path)
    X = df[FEATURE_COLUMNS].values.astype(np.float64)
    y = df[TARGET_COLUMN].values.astype(np.int64)
    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    train_df = pd.DataFrame(X_train_s, columns=FEATURE_COLUMNS)
    train_df[TARGET_COLUMN] = y_train
    test_df = pd.DataFrame(X_test_s, columns=FEATURE_COLUMNS)
    test_df[TARGET_COLUMN] = y_test
    train_raw = pd.DataFrame(X_train, columns=FEATURE_COLUMNS)
    train_raw[TARGET_COLUMN] = y_train
    return train_df, test_df, scaler, train_raw


def smote_resample(train_df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    X = train_df[FEATURE_COLUMNS].values
    y = train_df[TARGET_COLUMN].values
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)
    out = pd.DataFrame(X_res, columns=FEATURE_COLUMNS)
    out[TARGET_COLUMN] = y_res
    return out
