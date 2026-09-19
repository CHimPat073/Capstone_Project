"""
hybrid_retriever.py — Phase 2: Hybrid Retrieval & Fusion using EnsembleRetriever.

WHY HYBRID RETRIEVAL?
    - Dense (ChromaDB): Embeddings capture semantic context and paraphrase
      (e.g., matching "how to end the contract" with "Termination Clause").
    - Sparse (BM25): Frequency-based keyword matching excels at exact terms,
      defined entities, section numbers, and specialized legal terminology.
    - EnsembleRetriever: Combines both ranking lists using Reciprocal Rank
      Fusion (RRF), yielding higher hit-rate than either retriever alone.

WHY 0.5 / 0.5 WEIGHTS?
    An equal blend gives fair importance to both semantic meaning and exact
    lexical matches.

Configuration:
    - Dense retriever top-k: 20
    - BM25 retriever top-k:  20
    - Final retrieved top-k: 5
"""

from typing import Any, List, Sequence
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever


def build_hybrid_retriever(
    vector_retriever: Any,
    bm25_retriever: Any,
    weights: Sequence[float] = (0.5, 0.5),
) -> EnsembleRetriever:
    """
    Build an EnsembleRetriever combining dense vector and BM25 retrievers.

    Args:
        vector_retriever: Dense retriever (Chroma, k=20).
        bm25_retriever:   Keyword retriever (BM25, k=20).
        weights:          Tuple or list of float weights (default: [0.5, 0.5]).

    Returns:
        EnsembleRetriever instance.
    """
    if vector_retriever is None or bm25_retriever is None:
        raise ValueError("Both vector_retriever and bm25_retriever must be provided.")

    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=list(weights),
    )


def retrieve(query: str, index: Any, k: int = 5) -> List[Document]:
    """
    Execute hybrid retrieval for a query against a DocumentIndex.

    Args:
        query: Natural-language query string or search keywords.
        index: DocumentIndex containing the hybrid_retriever.
        k:     Number of top documents to return (default: 5).

    Returns:
        List of top-k Document objects with metadata preserved:
            - page_num
            - chunk_type ("text" or "table")
            - doc_id
            - parent_id
            - token_count
    """
    if not query or not query.strip():
        return []

    if not hasattr(index, "hybrid_retriever") or index.hybrid_retriever is None:
        raise ValueError("Index does not contain an active hybrid_retriever.")

    # EnsembleRetriever returns candidates ranked by fused RRF scores
    all_results = index.hybrid_retriever.invoke(query)

    # Slice to top-k
    return all_results[:k]
