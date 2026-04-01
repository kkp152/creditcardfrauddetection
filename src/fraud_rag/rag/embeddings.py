from __future__ import annotations

import os
from functools import lru_cache

from langchain_community.embeddings import HuggingFaceEmbeddings

from fraud_rag.config import DEFAULT_EMBED_MODEL, HF_CACHE_DIR

# Keep HF artifacts inside the project (portable, sandbox-friendly).
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR.parent / "hf_home"))
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=2)
def get_embeddings(model_name: str = DEFAULT_EMBED_MODEL) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        cache_folder=str(HF_CACHE_DIR),
    )
