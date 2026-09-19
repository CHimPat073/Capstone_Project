"""
chunker.py — Phase 1: Split text Documents into parent/child chunks.

WHY parent/child chunking?
    If we only store small chunks, we get precise keyword matches but lose
    surrounding context (the retriever fetches a 200-token snippet and the
    answer might span 800 tokens). If we only store large chunks, the embedding
    is diluted and similarity search becomes noisy.

    Solution: index small "child" chunks for search, but return their larger
    "parent" chunk as the actual context handed to the LLM (Phase 3).

HOW it works here:
    1. Every page Document is split into ~1 000-character "parent" chunks.
    2. Every parent chunk is split into ~800-character "child" chunks with overlap.
    3. Each child's metadata records its parent_id so Phase 2 can fetch the parent.
    4. Table Documents are NEVER split — they are returned as-is.

Character sizes (documented approximation, not exact token counts):
    PARENT_CHUNK_SIZE    = 1 000 chars ≈ 250 tokens
    CHILD_CHUNK_SIZE     =   800 chars ≈ 200 tokens
    CHILD_CHUNK_OVERLAP  =   200 chars ≈  50 tokens

We deliberately do NOT use a tokenizer here — adding a tokenizer would complicate
the pipeline for only a marginal accuracy gain at this stage.
"""

import uuid
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Chunk size constants (change these to tune retrieval granularity) ──────────
PARENT_CHUNK_SIZE = 1000    # characters; ~250 tokens (1 token ≈ 4 chars)
PARENT_CHUNK_OVERLAP = 100  # characters; keeps context across parent boundaries

CHILD_CHUNK_SIZE = 800      # characters; ~200 tokens — searchable unit
CHILD_CHUNK_OVERLAP = 200   # characters; prevents cutting answers in half
# ──────────────────────────────────────────────────────────────────────────────


def create_chunks(
    page_docs: List[Document],
    table_docs: List[Document],
) -> Tuple[List[Document], List[Document]]:
    """
    Split page Documents into parent and child chunks.
    Table Documents are passed through unchanged (never split).

    Args:
        page_docs:  Documents from pdf_loader (one per page, chunk_type="text").
        table_docs: Documents from table_extractor (chunk_type="table").

    Returns:
        (parent_chunks, child_chunks)
        parent_chunks — larger context chunks + unsplit table docs
        child_chunks  — smaller searchable chunks (tables already in parents too)

    Each child Document gets a 'parent_id' in its metadata pointing to its
    parent chunk's doc_id. The vector store indexes children; Phase 2 uses
    parent_id to look up full context.
    """

    # Splitters for parent and child levels
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
    )

    all_parents: List[Document] = []
    all_children: List[Document] = []

    # ── Text pages → parent chunks → child chunks ──────────────────────────
    for page_doc in page_docs:
        # Split one page into parent-level chunks
        parent_splits = parent_splitter.split_documents([page_doc])

        for parent_doc in parent_splits:
            # Give this parent a stable, unique ID
            parent_chunk_id = str(uuid.uuid4())
            parent_doc.metadata["chunk_id"] = parent_chunk_id
            parent_doc.metadata["chunk_type"] = "text"
            all_parents.append(parent_doc)

            # Split this parent into child chunks
            child_splits = child_splitter.split_documents([parent_doc])
            for child_doc in child_splits:
                child_doc.metadata["chunk_id"] = str(uuid.uuid4())
                child_doc.metadata["parent_id"] = parent_chunk_id  # link to parent
                child_doc.metadata["chunk_type"] = "text"
                # Recalculate token estimate for child's actual text
                child_doc.metadata["token_count"] = len(child_doc.page_content) // 4
                all_children.append(child_doc)

    # ── Table documents are NEVER split ───────────────────────────────────
    # Tables go into parent list directly; they also get their own child entry
    # so BM25 and vector search can find them through the child index.
    for table_doc in table_docs:
        table_chunk_id = str(uuid.uuid4())
        table_doc.metadata["chunk_id"] = table_chunk_id
        table_doc.metadata["chunk_type"] = "table"
        all_parents.append(table_doc)

        # The "child" of a table is the table itself — it is not split
        # We create a shallow copy so parent and child share the same text
        table_child = Document(
            page_content=table_doc.page_content,
            metadata={
                **table_doc.metadata,
                "chunk_id": str(uuid.uuid4()),
                "parent_id": table_chunk_id,   # points back to the table parent
                "chunk_type": "table",
            },
        )
        all_children.append(table_child)

    return all_parents, all_children
