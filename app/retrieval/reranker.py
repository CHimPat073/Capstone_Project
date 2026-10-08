"""Optional cross-encoder reranking for hybrid retrieval candidates."""

import os
from typing import Any, List, Optional

from langchain_core.documents import Document


class CrossEncoderReranker:
    """Rerank candidate passages with a local sentence-transformers model.

    The model is loaded lazily and can be disabled with RERANKER_ENABLED=false.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv(
            "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self._model = None

    def rerank(self, query: str, documents: List[Document], top_k: int) -> List[Document]:
        if not documents or os.getenv("RERANKER_ENABLED", "false").lower() in {"0", "false", "no"}:
            return documents[:top_k]
        try:
            if self._model is None:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name, device=os.getenv("RERANKER_DEVICE", "cpu"))
            scores = self._model.predict([(query, doc.page_content) for doc in documents])
            ranked = sorted(zip(scores, documents), key=lambda pair: float(pair[0]), reverse=True)
            return [doc for _, doc in ranked[:top_k]]
        except Exception:
            # Retrieval stays available if an optional model cannot be fetched or loaded.
            return documents[:top_k]


_DEFAULT_RERANKER = CrossEncoderReranker()


def rerank_documents(query: str, documents: List[Document], top_k: int = 5,
                      reranker: Optional[Any] = None) -> List[Document]:
    """Small injection-friendly wrapper used by retrieval and tests."""
    ranker = reranker or _DEFAULT_RERANKER
    return ranker.rerank(query, documents, top_k)
