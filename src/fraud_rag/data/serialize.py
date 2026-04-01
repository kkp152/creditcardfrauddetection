from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_rag.config import FEATURE_COLUMNS, TARGET_COLUMN


def _amount_bucket(amount: float) -> str:
    if amount < 10:
        return "very low"
    if amount < 50:
        return "low"
    if amount < 200:
        return "medium"
    if amount < 1000:
        return "high"
    return "very high"


def _time_bucket(t: float) -> str:
    # Time is seconds from first transaction; ~86400 per day
    hours = (t % 86400) / 3600
    if hours < 6:
        return "late night"
    if hours < 12:
        return "morning"
    if hours < 18:
        return "afternoon"
    return "evening"


def transaction_row_to_text(row: pd.Series | dict, fraud_label: bool | None = None) -> str:
    """
    Build a human-readable line for embedding / RAG.
    PCA features V* are summarized by magnitude (anomaly proxy).
    """
    if isinstance(row, dict):
        row = pd.Series(row)
    vals = [float(row[c]) for c in FEATURE_COLUMNS]
    time_s, amount = vals[0], vals[-1]
    v_feats = np.array(vals[1:-1], dtype=np.float64)
    v_abs_mean = float(np.mean(np.abs(v_feats)))
    v_max = float(np.max(np.abs(v_feats)))
    amt_b = _amount_bucket(amount)
    t_b = _time_bucket(time_s)
    parts = [
        f"Transaction amount {amount:.2f} ({amt_b} value),",
        f"time pattern {_time_bucket(time_s)} (hour context {t_b}),",
        f"PCA-derived behavior score mean {v_abs_mean:.3f} max {v_max:.3f}.",
    ]
    if fraud_label is True:
        parts.append("Label: confirmed fraud.")
    elif fraud_label is False:
        parts.append("Label: legitimate.")
    return " ".join(parts)


def transaction_to_text(features: dict[str, float], fraud_label: bool | None = None) -> str:
    row = {k: features[k] for k in FEATURE_COLUMNS}
    return transaction_row_to_text(row, fraud_label=fraud_label)


def dataframe_to_texts(df: pd.DataFrame, fraud_only: bool = True) -> list[str]:
    out: list[str] = []
    for _, row in df.iterrows():
        is_fraud = bool(row[TARGET_COLUMN]) if TARGET_COLUMN in row.index else None
        if fraud_only and not is_fraud:
            continue
        out.append(transaction_row_to_text(row, fraud_label=is_fraud if is_fraud is not None else True))
    return out
