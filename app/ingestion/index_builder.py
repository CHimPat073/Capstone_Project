"""
index_builder.py — Phase 1 & 2: Build the in-memory ChromaDB + BM25 hybrid index.

Call build_index(pdf_bytes) and receive a DocumentIndex containing:
    - .vector_retriever  (Chroma dense search, k=20)
    - .bm25_retriever    (BM25 lexical search, k=20)
    - .hybrid_retriever  (EnsembleRetriever combining dense + BM25)
    - .parent_chunks / .child_chunks / .page_docs

WHY EPHEMERAL CHROMADB?
    We are building a session-based assistant: one PDF in, answers out,
    no leftover data. Using chromadb.EphemeralClient() ensures:
    - Pure in-memory vector storage (RAM only)
    - No SQLite database files created
    - No persistence folders or disk traces
    - Complete disposal as soon as the index object goes out of scope

WHY BM25?
    Dense embeddings capture semantic intent but can miss precise alphanumeric
    identifiers (e.g. clause numbers, specific monetary figures, defined parties).
    BM25 fills this gap.

WHY ENSEMBLE HYBRID RETRIEVAL?
    LangChain's EnsembleRetriever merges results from both retrievers using
    Reciprocal Rank Fusion (RRF). Equal weighting [0.5, 0.5] balances semantic
    similarity with exact keyword matches.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, List

import chromadb
from langchain_chroma import Chroma
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning, message=r"`langchain-community` is being sunset")
    from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.table_extractor import extract_tables
from app.ingestion.chunker import create_chunks
from app.retrieval.hybrid_retriever import build_hybrid_retriever

# ── Embedding model (runs locally on CPU, no API key needed) ──────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# ──────────────────────────────────────────────────────────────────────────────

# Retrieval candidates pool size for each individual retriever
DENSE_TOP_K = 20
BM25_TOP_K = 20


@dataclass
class DocumentIndex:
    """
    Holds everything needed to perform retrieval.

    Deliberately simple — no base classes, no strategy patterns.
    """
    doc_id: str                          # unique ID for the uploaded PDF
    page_docs: List[Document]            # raw page-level documents (one per page)
    parent_chunks: List[Document]        # larger context chunks + unsplit tables
    child_chunks: List[Document]         # smaller searchable chunks (k=20)
    vector_retriever: Any                # Chroma dense retriever (k=20)
    bm25_retriever: BM25Retriever        # BM25 keyword retriever (k=20)
    hybrid_retriever: Any = None         # EnsembleRetriever (dense + BM25)
    num_pages: int = 0
    num_tables: int = 0


def _load_embeddings() -> HuggingFaceEmbeddings:
    """
    Load the HuggingFace embedding model.
    Runs on CPU; uses cosine similarity via normalize_embeddings=True.
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # cosine similarity
        )
        return embeddings
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {exc}"
        ) from exc


def build_chroma_store(
    child_chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
    doc_id: str,
) -> Chroma:
    """
    Build an in-memory ephemeral Chroma index from child chunk Documents.

    WHY Chroma EphemeralClient?
        Provides a standard vector database interface while keeping all collections
        and embeddings strictly in-memory (RAM only). Zero database files, zero
        disk writes, automatically cleared when the process or session ends.
    """
    if not child_chunks:
        raise ValueError("No child chunks to index — cannot build vector store.")

    # EphemeralClient ensures purely in-memory operation (no disk writes)
    client = chromadb.EphemeralClient()
    collection_name = f"doc_{doc_id.replace('-', '_')}"

    vector_store = Chroma.from_documents(
        documents=child_chunks,
        embedding=embeddings,
        client=client,
        collection_name=collection_name,
    )
    return vector_store


def build_bm25_retriever(child_chunks: List[Document], k: int = BM25_TOP_K) -> BM25Retriever:
    """
    Build an in-memory BM25 keyword retriever from child chunk Documents.

    BM25Retriever holds everything in memory — no external service,
    no disk storage.
    """
    if not child_chunks:
        raise ValueError("No child chunks to index — cannot build BM25 retriever.")

    retriever = BM25Retriever.from_documents(child_chunks)
    retriever.k = k
    return retriever


def build_index(pdf_bytes: bytes) -> DocumentIndex:
    """
    Full Phase 1 & 2 pipeline: PDF bytes → in-memory hybrid index.

    Steps:
        1. Generate a unique doc_id for this PDF
        2. Extract text pages with pypdf
        3. Extract tables with camelot (graceful fallback if unavailable)
        4. Split text into parent/child chunks (preserving metadata)
        5. Load the embedding model (MiniLM-L6-v2)
        6. Build an ephemeral ChromaDB vector store (k=20)
        7. Build an in-memory BM25 retriever (k=20)
        8. Combine them with LangChain EnsembleRetriever (weights 0.5/0.5)
        9. Return DocumentIndex dataclass

    Args:
        pdf_bytes: Raw bytes of the uploaded PDF.

    Returns:
        DocumentIndex — holds retrievers + metadata. Discard to free memory.
    """

    # ── Step 1: Unique document identifier ────────────────────────────────
    doc_id = str(uuid.uuid4())

    # ── Step 2: Extract text pages ────────────────────────────────────────
    page_docs = load_pdf(pdf_bytes, doc_id)
    num_pages = len(page_docs)

    # ── Step 3: Extract tables ────────────────────────────────────────────
    table_docs = extract_tables(pdf_bytes, doc_id)
    num_tables = len(table_docs)

    # ── Step 4: Create parent/child chunks ────────────────────────────────
    parent_chunks, child_chunks = create_chunks(page_docs, table_docs)

    if not child_chunks:
        raise ValueError(
            "No chunks were created. The PDF may contain only scanned images "
            "without extractable text."
        )

    # ── Step 5: Load embeddings ────────────────────────────────────────────
    embeddings = _load_embeddings()

    # ── Step 6: Build Ephemeral Chroma vector store ────────────────────────
    vector_store = build_chroma_store(child_chunks, embeddings, doc_id)
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": DENSE_TOP_K})

    # ── Step 7: Build BM25 retriever ──────────────────────────────────────
    bm25_retriever = build_bm25_retriever(child_chunks, k=BM25_TOP_K)

    # ── Step 8: Build Hybrid EnsembleRetriever (0.5 / 0.5) ─────────────────
    hybrid_retriever = build_hybrid_retriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        weights=(0.5, 0.5),
    )

    # ── Step 9: Return index object ───────────────────────────────────────
    return DocumentIndex(
        doc_id=doc_id,
        page_docs=page_docs,
        parent_chunks=parent_chunks,
        child_chunks=child_chunks,
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        hybrid_retriever=hybrid_retriever,
        num_pages=num_pages,
        num_tables=num_tables,
    )
