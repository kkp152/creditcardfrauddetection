from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from fraud_rag.config import ARTIFACTS_DIR, FEATURE_COLUMNS, TARGET_COLUMN

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional lib / OpenMP
    XGBClassifier = None


def _metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
    }


def train_baselines(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path | None = None,
) -> dict[str, dict]:
    out_dir = out_dir or ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df[TARGET_COLUMN].values
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df[TARGET_COLUMN].values

    models: dict[str, object] = {
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    }
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="logloss",
        )
    else:
        models["hist_gradient_boosting"] = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.05,
            max_iter=300,
            random_state=42,
        )

    report: dict[str, dict] = {}
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        if hasattr(clf, "predict_proba"):
            prob = clf.predict_proba(X_test)[:, 1]
        else:
            prob = clf.predict(X_test).astype(float)
        m = _metrics(y_test, prob)
        report[name] = m
        joblib.dump(clf, out_dir / f"{name}.joblib")
        with open(out_dir / f"{name}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2)

    with open(out_dir / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report
