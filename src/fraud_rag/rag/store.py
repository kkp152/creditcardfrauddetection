from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from fraud_rag.config import DEFAULT_EMBED_MODEL
from fraud_rag.rag.embeddings import get_embeddings


def build_faiss_index(texts: list[str], out_dir: Path, model_name: str | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    emb = get_embeddings(model_name or DEFAULT_EMBED_MODEL)
    docs = [Document(page_content=t, metadata={"idx": i}) for i, t in enumerate(texts)]
    vs = FAISS.from_documents(docs, emb)
    vs.save_local(str(out_dir))
    return out_dir


def load_faiss_index(index_dir: Path, model_name: str | None = None) -> FAISS:
    emb = get_embeddings(model_name or DEFAULT_EMBED_MODEL)
    return FAISS.load_local(
        str(index_dir),
        emb,
        allow_dangerous_deserialization=True,
    )


def faiss_direct_search(
    index_dir: Path,
    query_text: str,
    k: int = 3,
    model_name: str | None = None,
) -> list[str]:
    """Lightweight search without full LangChain chain (for tests)."""
    vs = load_faiss_index(index_dir, model_name=model_name or DEFAULT_EMBED_MODEL)
    pairs = vs.similarity_search(query_text, k=k)
    return [d.page_content for d in pairs]
