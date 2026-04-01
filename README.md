# Fraud Detection & Explainability (RAG)

End-to-end pipeline: **SMOTE-balanced baselines** (Random Forest, XGBoost or HistGradientBoosting), **PyTorch tabular MLP**, **FastAPI** serving, and a **FAISS + LangChain** retrieval layer for natural-language-style explanations over embedded fraud narratives.

## Setup

```bash
cd fraud-rag-platform
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) `creditcard.csv` under `data/creditcard.csv`.

## Train

```bash
python train.py
# Optional: --hf  (DistilBERT on serialized transactions; slow)
# Optional: --skip-rag
# Optional: --epochs-mlp 30
```

Artifacts are written to `artifacts/` (scaler, models, `train_meta.json`, `faiss_index/`). Embeddings use a project-local cache at `.hf_cache/`.

**XGBoost:** On macOS, install OpenMP (`brew install libomp`) or rely on the automatic fallback to `HistGradientBoostingClassifier`.

## API

```bash
export PYTHONPATH=src
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- `GET /health` — model and RAG status  
- `POST /predict` — single transaction (JSON body with `Time`, `V1`–`V28`, `Amount`)  
- `POST /batch-predict` — list of transactions  
- `POST /batch-predict-csv` — CSV upload with the same columns  
- `POST /explain` — prediction + RAG explanation + similar past cases  

## Streamlit

```bash
export PYTHONPATH=src
streamlit run streamlit_app.py
```

## Layout

- `src/fraud_rag/` — data prep, models, inference, RAG  
- `app/main.py` — FastAPI  
- `train.py` — training + FAISS index build  

## App Deployed

- `URL:` - https://credit-card-cc-fraud-detection.streamlit.app/
