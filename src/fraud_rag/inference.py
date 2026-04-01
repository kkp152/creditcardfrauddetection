from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import torch

from fraud_rag.config import ARTIFACTS_DIR, FEATURE_COLUMNS
from fraud_rag.models.pytorch_model import TabularMLP, load_mlp


ModelKind = Literal["mlp", "random_forest", "xgboost", "hist_gradient_boosting", "hf"]


def _top_features_rf(model, feature_names: list[str], k: int = 5) -> list[tuple[str, float]]:
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return []
    pairs = sorted(zip(feature_names, imp.tolist()), key=lambda x: -x[1])[:k]
    return pairs


def _top_features_mlp(model: TabularMLP, x: np.ndarray, device: str, k: int = 5) -> list[tuple[str, float]]:
    model.eval()
    xt = torch.tensor(x, dtype=torch.float32, device=device, requires_grad=True)
    logit = model(xt.unsqueeze(0)).squeeze()
    prob = torch.sigmoid(logit)
    prob.backward()
    grad = xt.grad.detach().cpu().numpy()
    contrib = np.abs(grad * x)
    pairs = sorted(zip(FEATURE_COLUMNS, contrib.tolist()), key=lambda x: -x[1])[:k]
    return pairs


class FraudInference:
    """Loads scaler + one of sklearn / PyTorch / HF artifacts."""

    def __init__(
        self,
        artifacts_dir: Path | None = None,
        backend: ModelKind = "mlp",
    ):
        self.artifacts_dir = Path(artifacts_dir or ARTIFACTS_DIR)
        self.backend = backend
        self.scaler = None
        self._sk_model = None
        self._mlp: TabularMLP | None = None
        self._mlp_ckpt: dict | None = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._hf_model = None
        self._hf_tokenizer = None
        self._load()

    def _load(self) -> None:
        meta_path = self.artifacts_dir / "train_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.backend = meta.get("active_backend", self.backend)

        scaler_path = self.artifacts_dir / "scaler.joblib"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)

        def _try_mlp() -> bool:
            p = self.artifacts_dir / "mlp.pt"
            if not p.exists():
                return False
            self._mlp, self._mlp_ckpt = load_mlp(p, self._device)
            return True

        def _try_rf() -> bool:
            p = self.artifacts_dir / "random_forest.joblib"
            if not p.exists():
                return False
            self._sk_model = joblib.load(p)
            return True

        def _try_xgb() -> bool:
            p = self.artifacts_dir / "xgboost.joblib"
            if not p.exists():
                return False
            self._sk_model = joblib.load(p)
            return True

        def _try_hist_gb() -> bool:
            p = self.artifacts_dir / "hist_gradient_boosting.joblib"
            if not p.exists():
                return False
            self._sk_model = joblib.load(p)
            return True

        def _try_hf() -> bool:
            p = self.artifacts_dir / "hf_classifier"
            if not p.exists():
                return False
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._hf_tokenizer = AutoTokenizer.from_pretrained(p)
            self._hf_model = AutoModelForSequenceClassification.from_pretrained(p)
            self._hf_model.to(self._device)
            self._hf_model.eval()
            return True

        order = {
            "mlp": _try_mlp,
            "random_forest": _try_rf,
            "xgboost": _try_xgb,
            "hist_gradient_boosting": _try_hist_gb,
            "hf": _try_hf,
        }
        primary = order.get(self.backend, _try_mlp)
        if not primary():
            for name, fn in (
                ("mlp", _try_mlp),
                ("random_forest", _try_rf),
                ("xgboost", _try_xgb),
                ("hist_gradient_boosting", _try_hist_gb),
                ("hf", _try_hf),
            ):
                if fn():
                    self.backend = name  # type: ignore[assignment]
                    break

    @property
    def ready(self) -> bool:
        return (
            self.scaler is not None
            and (
                self._mlp is not None
                or self._sk_model is not None
                or self._hf_model is not None
            )
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return X
        return self.scaler.transform(X)

    def predict_proba_row(self, features: np.ndarray) -> float:
        """features: shape (30,) raw or scaled depending on caller — we expect RAW from API."""
        x = features.reshape(1, -1)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        x = x.astype(np.float32)

        if self._mlp is not None:
            with torch.no_grad():
                t = torch.from_numpy(x).to(self._device)
                p = torch.sigmoid(self._mlp(t)).item()
            return float(p)
        if self._sk_model is not None:
            if hasattr(self._sk_model, "predict_proba"):
                return float(self._sk_model.predict_proba(x)[0, 1])
            return float(self._sk_model.predict(x)[0])
        if self._hf_model is not None:
            from fraud_rag.data.serialize import transaction_to_text

            flat = np.asarray(features, dtype=np.float64).ravel()
            d = {FEATURE_COLUMNS[i]: float(flat[i]) for i in range(len(FEATURE_COLUMNS))}
            text = transaction_to_text(d)
            enc = self._hf_tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt",
            )
            enc = {k: v.to(self._device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self._hf_model(**enc).logits
                p = torch.softmax(logits, dim=-1)[0, 1].item()
            return float(p)
        raise RuntimeError("No model loaded")

    def top_features(self, raw_features: np.ndarray, k: int = 5) -> list[dict[str, Any]]:
        x = raw_features.reshape(1, -1).astype(np.float32)
        if self.scaler is not None:
            xs = self.scaler.transform(x).astype(np.float32)
        else:
            xs = x

        if self._mlp is not None:
            pairs = _top_features_mlp(self._mlp, xs.flatten(), self._device, k=k)
            return [{"name": n, "weight": float(w)} for n, w in pairs]
        if self._sk_model is not None:
            pairs = _top_features_rf(self._sk_model, FEATURE_COLUMNS, k=k)
            return [{"name": n, "weight": float(w)} for n, w in pairs]
        # HF: fall back to unscaled gradient not defined — return empty or PCA proxy
        return [{"name": "Amount", "weight": 1.0}, {"name": "Time", "weight": 0.5}][:k]
