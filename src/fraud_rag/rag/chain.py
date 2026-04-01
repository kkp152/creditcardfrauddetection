from __future__ import annotations

from langchain_core.runnables import RunnableLambda

from fraud_rag.rag.store import load_faiss_index


def _format_explanation(query: str, docs: list, k: int) -> str:
    if not docs:
        return (
            "No sufficiently similar historical fraud narratives were retrieved; "
            "the score still exceeded the model threshold based on tabular signals alone."
        )
    snippets = "; ".join(d.page_content[:240] for d in docs[:k])
    return (
        f"This case aligns with {len(docs)} prior fraud-relevant patterns in embedded history. "
        f"Similar contexts include: {snippets}"
    )


def build_explain_runnable(index_dir, k: int = 3, model_name: str | None = None):
    store = load_faiss_index(index_dir, model_name=model_name)

    def _invoke(query: str) -> dict:
        docs = store.similarity_search(query, k=k)
        return {
            "explanation": _format_explanation(query, docs, k),
            "similar_cases": [d.page_content for d in docs],
        }

    return RunnableLambda(_invoke)
