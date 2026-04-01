#!/usr/bin/env python3
"""Streamlit UI: paste features or upload CSV, view prediction + RAG explanation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fraud_rag.config import ARTIFACTS_DIR, FEATURE_COLUMNS
from fraud_rag.data.serialize import transaction_to_text
from fraud_rag.inference import FraudInference
from fraud_rag.rag.chain import build_explain_runnable

st.set_page_config(page_title="Fraud RAG Demo", layout="wide")
st.title("Fraud detection + RAG explainability")

if "inf" not in st.session_state:
    st.session_state.inf = FraudInference(ARTIFACTS_DIR)
if "rag" not in st.session_state:
    p = ARTIFACTS_DIR / "faiss_index"
    st.session_state.rag = build_explain_runnable(p) if p.exists() else None

inf = st.session_state.inf
rag = st.session_state.rag

col1, col2 = st.columns(2)
with col1:
    st.subheader("Transaction features")
    defaults = {c: 0.0 for c in FEATURE_COLUMNS}
    defaults["Amount"] = 50.0
    defaults["Time"] = 1000.0
    inputs = {}
    for c in FEATURE_COLUMNS:
        inputs[c] = st.number_input(c, value=float(defaults.get(c, 0.0)), format="%.6f")

with col2:
    st.subheader("Prediction")
    if not inf.ready:
        st.warning("Train the model first: `python train.py` with `data/creditcard.csv`.")
    else:
        raw = np.array([inputs[c] for c in FEATURE_COLUMNS], dtype=np.float64)
        p = inf.predict_proba_row(raw)
        st.metric("Fraud probability", f"{p:.4f}")
        st.write("Label:", "fraud" if p >= 0.5 else "legitimate")
        st.write("Top features:", inf.top_features(raw, k=5))

st.subheader("RAG explanation")
if rag is None:
    st.info("FAISS index not found. Run `python train.py` without `--skip-rag`.")
else:
    q = transaction_to_text({c: inputs[c] for c in FEATURE_COLUMNS})
    out = rag.invoke(q)
    st.write(out["explanation"])
    with st.expander("Similar embedded cases"):
        for i, s in enumerate(out["similar_cases"], 1):
            st.markdown(f"{i}. {s}")

st.subheader("Batch CSV")
up = st.file_uploader("creditcard-shaped CSV", type=["csv"])
if up and inf.ready:
    df = pd.read_csv(up)
    if set(FEATURE_COLUMNS).issubset(df.columns):
        for j, row in df.head(20).iterrows():
            raw = np.array([float(row[c]) for c in FEATURE_COLUMNS], dtype=np.float64)
            pr = inf.predict_proba_row(raw)
            st.write(f"Row {j}: p={pr:.4f}")
    else:
        st.error("Missing feature columns.")
