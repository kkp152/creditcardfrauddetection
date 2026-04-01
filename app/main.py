from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fraud_rag.config import ARTIFACTS_DIR, FEATURE_COLUMNS
from fraud_rag.data.serialize import transaction_to_text
from fraud_rag.inference import FraudInference
from fraud_rag.rag.chain import build_explain_runnable
from fraud_rag.schemas import (
    BatchPredictResponse,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    PredictResponse,
    TransactionInput,
)

app = FastAPI(title="Fraud Detection & RAG Explainability", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_inference: FraudInference | None = None
_explain = None


def get_inference() -> FraudInference:
    global _inference
    if _inference is None:
        _inference = FraudInference(ARTIFACTS_DIR)
    return _inference


def get_explain():
    global _explain
    if _explain is None:
        idx = ARTIFACTS_DIR / "faiss_index"
        if idx.exists():
            _explain = build_explain_runnable(idx)
    return _explain


def _predict_one(tx: TransactionInput) -> PredictResponse:
    inf = get_inference()
    if not inf.ready:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts missing. Run: python train.py (with data/creditcard.csv).",
        )
    raw = np.array(tx.to_feature_list(), dtype=np.float64)
    p = inf.predict_proba_row(raw)
    label = "fraud" if p >= 0.5 else "legitimate"
    tops = inf.top_features(raw, k=5)
    return PredictResponse(
        fraud_probability=p,
        label=label,
        model_name=inf.backend,
        top_features=tops,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    inf = get_inference()
    ex = get_explain()
    return HealthResponse(
        status="ok",
        model_loaded=inf.ready,
        rag_ready=ex is not None,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(tx: TransactionInput):
    return _predict_one(tx)


@app.post("/batch-predict", response_model=BatchPredictResponse)
async def batch_predict(transactions: list[TransactionInput]):
    results = [_predict_one(t) for t in transactions]
    return BatchPredictResponse(results=results)


@app.post("/batch-predict-csv", response_model=BatchPredictResponse)
async def batch_predict_csv(file: UploadFile = File(...)):
    raw = await file.read()
    df = pd.read_csv(io.BytesIO(raw))
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise HTTPException(400, f"CSV missing columns: {missing}")
    txs = []
    for _, row in df.iterrows():
        d = {c: float(row[c]) for c in FEATURE_COLUMNS}
        txs.append(TransactionInput(**d))
    return BatchPredictResponse(results=[_predict_one(t) for t in txs])


@app.post("/explain", response_model=ExplainResponse)
async def explain(req: ExplainRequest):
    pred = _predict_one(req.transaction)
    ex = get_explain()
    if ex is None:
        return ExplainResponse(
            prediction=pred,
            explanation="RAG index not built. Run train.py without --skip-rag.",
            similar_cases=[],
        )
    q = transaction_to_text(req.transaction.to_feature_dict())
    out = ex.invoke(q)
    return ExplainResponse(
        prediction=pred,
        explanation=out["explanation"],
        similar_cases=out["similar_cases"][: req.top_k],
    )
