#!/usr/bin/env python3
"""Train baselines, PyTorch MLP, build FAISS RAG index. Requires data/creditcard.csv."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fraud_rag.config import ARTIFACTS_DIR, DATA_DIR, TARGET_COLUMN
from fraud_rag.data.preprocess import load_and_split, smote_resample
from fraud_rag.data.serialize import dataframe_to_texts
from fraud_rag.models.baseline import train_baselines
from fraud_rag.models.hf_classifier import train_hf_classifier
from fraud_rag.models.pytorch_model import train_mlp
from fraud_rag.rag.store import build_faiss_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_DIR / "creditcard.csv")
    parser.add_argument("--out", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--epochs-mlp", type=int, default=30)
    parser.add_argument("--hf", action="store_true", help="Fine-tune DistilBERT on serialized transactions (slow)")
    parser.add_argument("--skip-rag", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    train_df, test_df, scaler, train_raw = load_and_split(path=args.data)
    joblib.dump(scaler, args.out / "scaler.joblib")

    train_bal = smote_resample(train_df)
    print("Training sklearn baselines...")
    baseline_report = train_baselines(train_bal, test_df, out_dir=args.out)
    print("Training PyTorch MLP...")
    mlp_metrics = train_mlp(train_bal, test_df, epochs=args.epochs_mlp, out_dir=args.out)

    active = "mlp"
    best_f1 = mlp_metrics.get("f1", 0.0)
    for name, m in baseline_report.items():
        if m.get("f1", 0) > best_f1:
            best_f1 = m["f1"]
            active = name

    meta = {
        "active_backend": active,
        "baseline_metrics": baseline_report,
        "mlp_metrics": mlp_metrics,
    }

    if args.hf:
        print("Fine-tuning HuggingFace classifier (this may take a while)...")
        hf_m = train_hf_classifier(train_bal, test_df, out_dir=args.out)
        meta["hf_metrics"] = hf_m
        if hf_m.get("f1", 0) > best_f1:
            meta["active_backend"] = "hf"

    with open(args.out / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if not args.skip_rag:
        fraud_raw = train_raw[train_raw[TARGET_COLUMN] == 1]
        texts = dataframe_to_texts(fraud_raw, fraud_only=False)
        if len(texts) < 10:
            texts.extend(texts * 5)
        print(f"Building FAISS index with {len(texts)} fraud narratives...")
        build_faiss_index(texts, args.out / "faiss_index")
        meta["rag_documents"] = len(texts)
        with open(args.out / "train_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    print("Done. Metrics:", json.dumps(meta, indent=2)[:2000])


if __name__ == "__main__":
    main()
