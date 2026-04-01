"""
Optional HuggingFace DistilBERT classifier on serialized transaction text.
Slower to train; use train.py --hf for a short fine-tuning run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from fraud_rag.config import ARTIFACTS_DIR, DEFAULT_HF_CLASSIFIER, TARGET_COLUMN
from fraud_rag.data.serialize import transaction_row_to_text


class _TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 128):
        self.enc = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        return {k: v[i] for k, v in self.enc.items()}, self.labels[i]


def train_hf_classifier(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = DEFAULT_HF_CLASSIFIER,
    epochs: int = 2,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_length: int = 128,
    out_dir: Path | None = None,
) -> dict[str, float]:
    out_dir = out_dir or ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    texts_tr = [transaction_row_to_text(row) for _, row in train_df.iterrows()]
    y_tr = train_df[TARGET_COLUMN].astype(int).tolist()
    texts_te = [transaction_row_to_text(row) for _, row in test_df.iterrows()]
    y_te = test_df[TARGET_COLUMN].astype(int).tolist()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    ds_tr = _TextDataset(texts_tr, y_tr, tokenizer, max_length=max_length)
    ds_te = _TextDataset(texts_te, y_te, tokenizer, max_length=max_length)
    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    loader_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(loader_tr) * epochs
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0], device=device))

    model.train()
    step = 0
    for _ in range(epochs):
        for batch, labels in loader_tr:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = labels.to(device)
            opt.zero_grad()
            logits = model(**batch).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1

    model.eval()
    probs: list[float] = []
    with torch.no_grad():
        for batch, _ in loader_te:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            p = torch.softmax(logits, dim=-1)[:, 1]
            probs.extend(p.cpu().numpy().tolist())

    y_prob = np.array(probs, dtype=np.float64)
    y_true = np.array(y_te, dtype=np.int64)
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
    }

    save_path = out_dir / "hf_classifier"
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    with open(save_path / "hf_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
