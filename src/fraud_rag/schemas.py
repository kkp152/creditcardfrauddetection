from typing import Any

from pydantic import BaseModel, Field

from fraud_rag.config import FEATURE_COLUMNS


class TransactionInput(BaseModel):
    """Single transaction feature vector (Kaggle ULB credit card schema)."""

    Time: float = Field(..., description="Seconds elapsed between transaction and first in dataset")
    Amount: float = Field(..., ge=0, description="Transaction amount")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    def to_feature_dict(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in FEATURE_COLUMNS}

    def to_feature_list(self) -> list[float]:
        return [getattr(self, k) for k in FEATURE_COLUMNS]


class PredictResponse(BaseModel):
    fraud_probability: float
    label: str
    model_name: str
    top_features: list[dict[str, Any]]


class BatchPredictRequest(BaseModel):
    transactions: list[TransactionInput]


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]


class ExplainRequest(BaseModel):
    transaction: TransactionInput
    top_k: int = Field(3, ge=1, le=20)


class ExplainResponse(BaseModel):
    prediction: PredictResponse
    explanation: str
    similar_cases: list[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    rag_ready: bool
