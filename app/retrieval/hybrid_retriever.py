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

import re
from typing import Any, List, Sequence
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever
from app.retrieval.reranker import rerank_documents


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


def retrieve(query: str, index: Any, k: int = 5, mode: str = "hybrid") -> List[Document]:
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

    if mode not in {"hybrid", "semantic", "exact"}:
        raise ValueError("mode must be 'hybrid', 'semantic', or 'exact'.")
    retriever = {
        "hybrid": getattr(index, "hybrid_retriever", None),
        "semantic": getattr(index, "vector_retriever", None),
        "exact": getattr(index, "bm25_retriever", None),
    }[mode]
    if retriever is None:
        raise ValueError("Index does not contain an active hybrid_retriever.")

    # Dense models can miss short paraphrases such as "intended to achieve".
    # Add a small, deterministic intent expansion while keeping the selected
    # retrieval mode intact. This improves semantic recall without changing the
    # user's requested exact or hybrid behavior.
    search_query = query
    if mode == "semantic":
        lowered = query.lower()
        if re.search(r"\b(intended to achieve|aim|purpose|objective|goal)\b", lowered):
            search_query = f"{query} objectives goals purpose scope"

    # EnsembleRetriever returns candidates ranked by fused RRF scores
    all_results = retriever.invoke(search_query)

    # Slice to top-k
    candidate_count = min(len(all_results), max(k * 3, k)) if mode == "hybrid" else min(len(all_results), k)
    reranker = getattr(index, "reranker", None)
    if mode == "hybrid":
        return rerank_documents(query, all_results[:candidate_count], k, reranker=reranker)
    return all_results[:k]
