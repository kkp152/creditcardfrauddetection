from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from fraud_rag.config import ARTIFACTS_DIR, FEATURE_COLUMNS, TARGET_COLUMN


class TabularMLP(nn.Module):
    """PyTorch classifier for standardized tabular fraud features."""

    def __init__(self, input_dim: int, hidden: tuple[int, ...] = (256, 128, 64), dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _metrics_np(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, prob)),
    }


def train_mlp(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    epochs: int = 30,
    batch_size: int = 2048,
    lr: float = 1e-3,
    device: str | None = None,
    out_dir: Path | None = None,
) -> dict[str, float]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = out_dir or ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train = train_df[FEATURE_COLUMNS].values.astype(np.float32)
    y_train = train_df[TARGET_COLUMN].values.astype(np.float32)
    X_test = test_df[FEATURE_COLUMNS].values.astype(np.float32)
    y_test = test_df[TARGET_COLUMN].values.astype(np.float32)

    ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = TabularMLP(input_dim=X_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    pos_w = max(1.0, n_neg / max(n_pos, 1.0))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.from_numpy(X_test).to(device))).cpu().numpy()
    metrics = _metrics_np(y_test, prob)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": X_train.shape[1],
            "hidden": (256, 128, 64),
        },
        out_dir / "mlp.pt",
    )
    with open(out_dir / "mlp_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def load_mlp(path: Path, device: str | None = None) -> tuple[TabularMLP, dict]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device)
    model = TabularMLP(input_dim=ckpt["input_dim"], hidden=tuple(ckpt["hidden"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt
