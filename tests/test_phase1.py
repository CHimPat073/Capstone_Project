"""
test_phase1.py — Phase 1 verification test.

Runs automatically with:   pytest tests/test_phase1.py -v
Or directly with:          python tests/test_phase1.py

What this tests:
    - PDF loads successfully from bytes
    - Pages are extracted with correct page numbers
    - Documents have required metadata fields
    - Tables are detected and kept as single documents (not split)
    - Parent/child chunks are created with parent_id links
    - Embedding model loads and produces embeddings
    - FAISS vector index is created in memory only
    - BM25 keyword index is created
    - No database files or vector store folders appear on disk
    - Both retrievers return results for a real query
"""

import os
import sys
import io
import time
import glob

import pytest

# ── Make sure we can import from the project root ─────────────────────────────
# When running from the project root, 'app' is already importable.
# This sys.path line handles running the script directly from inside tests/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.table_extractor import extract_tables
from app.ingestion.chunker import create_chunks, CHILD_CHUNK_SIZE, PARENT_CHUNK_SIZE
from app.ingestion.index_builder import build_index, DocumentIndex

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_minimal_pdf() -> bytes:
    """
    Create the smallest valid single-page PDF in pure Python.
    No external file needed — good for CI and quick local runs.
    The PDF contains the text 'Hello Phase 1 Test'.
    """
    # Minimal valid PDF structure (hand-crafted, ~500 bytes)
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj

2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj

3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj

4 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td (Hello Phase 1 Test) Tj ET
endstream
endobj

5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj

xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000266 00000 n 
0000000360 00000 n 

trailer
<< /Size 6 /Root 1 0 R >>
startxref
441
%%EOF"""
    return pdf_content


def _get_sample_pdf_path() -> str | None:
    """
    Look for any PDF in the project directory to use as a real-world test.
    Returns the path to the first PDF found, or None if none exist.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    patterns = [
        os.path.join(project_root, "*.pdf"),
        os.path.join(project_root, "tests", "*.pdf"),
        os.path.join(project_root, "tests", "data", "*.pdf"),
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None


# ── Helper to print a separator ───────────────────────────────────────────────
def _section(title: str):
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}")


# ==================================================================
#  UNIT TESTS (fast, no embedding model needed)
# ==================================================================


class TestPdfLoader:
    """Tests for pdf_loader.load_pdf()"""

    def test_load_minimal_pdf(self):
        """PDF loads without error and returns at least one Document."""
        pdf_bytes = _make_minimal_pdf()
        docs = load_pdf(pdf_bytes, doc_id="test-doc")
        assert len(docs) >= 1, "Should return at least one Document"

    def test_page_numbers_present(self):
        """Every Document has a 1-based page_num in metadata."""
        pdf_bytes = _make_minimal_pdf()
        docs = load_pdf(pdf_bytes, doc_id="test-doc")
        for doc in docs:
            assert "page_num" in doc.metadata
            assert doc.metadata["page_num"] >= 1

    def test_doc_id_propagated(self):
        """The doc_id we pass in appears in every Document's metadata."""
        pdf_bytes = _make_minimal_pdf()
        docs = load_pdf(pdf_bytes, doc_id="my-unique-id")
        for doc in docs:
            assert doc.metadata["doc_id"] == "my-unique-id"

    def test_chunk_type_is_text(self):
        """Raw page documents should have chunk_type='text'."""
        pdf_bytes = _make_minimal_pdf()
        docs = load_pdf(pdf_bytes, doc_id="test-doc")
        for doc in docs:
            assert doc.metadata["chunk_type"] == "text"

    def test_required_metadata_fields(self):
        """All required metadata fields exist."""
        required = {"doc_id", "page_num", "chunk_type", "parent_id", "token_count"}
        pdf_bytes = _make_minimal_pdf()
        docs = load_pdf(pdf_bytes, doc_id="test-doc")
        for doc in docs:
            missing = required - set(doc.metadata.keys())
            assert not missing, f"Missing metadata fields: {missing}"

    def test_empty_bytes_raises(self):
        """Empty bytes should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            load_pdf(b"", doc_id="test")

    def test_invalid_pdf_raises(self):
        """Garbage bytes should raise ValueError."""
        with pytest.raises(ValueError):
            load_pdf(b"this is not a pdf", doc_id="test")


class TestChunker:
    """Tests for chunker.create_chunks()"""

    def _page_doc(self, text: str = None, page_num: int = 1) -> "Document":
        from langchain_core.documents import Document
        return Document(
            page_content=text or ("Word " * 500),  # ~500 words ≈ 2500 chars
            metadata={
                "doc_id": "test",
                "page_num": page_num,
                "chunk_type": "text",
                "parent_id": "",
                "token_count": 500,
            },
        )

    def _table_doc(self) -> "Document":
        from langchain_core.documents import Document
        return Document(
            page_content="| A | B |\n|---|---|\n| 1 | 2 |",
            metadata={
                "doc_id": "test",
                "page_num": 1,
                "chunk_type": "table",
                "parent_id": "",
                "token_count": 10,
            },
        )

    def test_child_chunks_created(self):
        """Text pages should produce child chunks."""
        parents, children = create_chunks([self._page_doc()], [])
        assert len(children) > 0, "Should produce at least one child chunk"

    def test_parent_id_set_on_children(self):
        """Each child should have a non-empty parent_id."""
        parents, children = create_chunks([self._page_doc()], [])
        for child in children:
            assert child.metadata.get("parent_id"), \
                "child.metadata['parent_id'] must point to a parent chunk"

    def test_parent_id_links_to_real_parent(self):
        """parent_id on each child must match a parent chunk's chunk_id."""
        parents, children = create_chunks([self._page_doc()], [])
        parent_ids = {p.metadata["chunk_id"] for p in parents}
        for child in children:
            assert child.metadata["parent_id"] in parent_ids, \
                "child's parent_id should match an actual parent chunk"

    def test_child_chunks_not_larger_than_child_size(self):
        """Child chunks should respect CHILD_CHUNK_SIZE."""
        parents, children = create_chunks([self._page_doc()], [])
        text_children = [c for c in children if c.metadata["chunk_type"] == "text"]
        for child in text_children:
            # Allow a small tolerance for overlap
            assert len(child.page_content) <= CHILD_CHUNK_SIZE + 50, \
                f"Child chunk too large: {len(child.page_content)}"

    def test_table_not_split(self):
        """A table document should appear as a single unit in parents."""
        table = self._table_doc()
        parents, children = create_chunks([], [table])

        # The table should appear exactly once in parents
        table_parents = [p for p in parents if p.metadata["chunk_type"] == "table"]
        assert len(table_parents) == 1, "Table should be exactly one parent chunk"

        # The child of a table should have the same content as the table
        table_children = [c for c in children if c.metadata["chunk_type"] == "table"]
        assert len(table_children) == 1
        assert table_children[0].page_content == table.page_content, \
            "Table child must have identical content to the table (not split)"


# ==================================================================
#  INTEGRATION TESTS (slower -- loads embedding model)
# ==================================================================


class TestBuildIndex:
    """
    Integration tests for build_index().
    These tests load the real embedding model (~90 MB download on first run).
    """

    @pytest.fixture(scope="class")
    def index(self, request):
        """Build index once and share across all tests in this class."""
        pdf_bytes = _make_minimal_pdf()
        idx = build_index(pdf_bytes)
        request.cls.idx = idx
        return idx


    def test_returns_document_index(self, index):
        """build_index should return a DocumentIndex."""
        assert isinstance(index, DocumentIndex)

    def test_doc_id_is_uuid(self, index):
        """doc_id should be a non-empty string (UUID)."""
        import uuid
        assert index.doc_id
        uuid.UUID(index.doc_id)  # raises if not valid UUID

    def test_page_docs_created(self, index):
        """At least one page document should exist."""
        assert len(index.page_docs) >= 1

    def test_child_chunks_created(self, index):
        """At least one child chunk should exist."""
        assert len(index.child_chunks) >= 1

    def test_parent_chunks_created(self, index):
        """At least one parent chunk should exist."""
        assert len(index.parent_chunks) >= 1

    def test_vector_retriever_exists(self, index):
        """vector_retriever should be a non-None object."""
        assert index.vector_retriever is not None

    def test_bm25_retriever_exists(self, index):
        """bm25_retriever should be a BM25Retriever."""
        from langchain_community.retrievers import BM25Retriever
        assert isinstance(index.bm25_retriever, BM25Retriever)

    def test_vector_retriever_returns_results(self, index):
        """Vector retriever should return at least one Document for a query."""
        results = index.vector_retriever.invoke("Hello Phase 1 Test")
        assert len(results) >= 1, "Vector retriever returned no results"

    def test_bm25_retriever_returns_results(self, index):
        """BM25 retriever should return at least one Document for a query."""
        results = index.bm25_retriever.invoke("Hello")
        assert len(results) >= 1, "BM25 retriever returned no results"

    def test_no_faiss_files_on_disk(self, index):
        """FAISS should NOT have written any index files to disk."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        faiss_files = glob.glob(os.path.join(project_root, "**", "*.faiss"), recursive=True)
        faiss_index_files = glob.glob(os.path.join(project_root, "**", "index.faiss"), recursive=True)
        assert not faiss_files and not faiss_index_files, \
            f"FAISS index files found on disk: {faiss_files}"

    def test_no_chroma_dir_on_disk(self, index):
        """No ChromaDB or vector store directories should exist."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        chroma_dirs = glob.glob(os.path.join(project_root, "**", "chroma*"), recursive=True)
        # Exclude Python package files
        chroma_dirs = [d for d in chroma_dirs if "site-packages" not in d]
        assert not chroma_dirs, f"Chroma directories found on disk: {chroma_dirs}"

    def test_metadata_page_num_survives(self, index):
        """page_num metadata should survive from raw pages through to child chunks."""
        for child in index.child_chunks:
            assert "page_num" in child.metadata
            assert child.metadata["page_num"] >= 1


# ==================================================================
#  HUMAN-READABLE DEMO (run directly with: python tests/test_phase1.py)
# ==================================================================


def run_demo():
    """
    Print a human-readable Phase 1 status report.
    Run with: python tests/test_phase1.py
    """
    results = {}

    print("\n" + "=" * 60)
    print("  DOCUMENT INTELLIGENCE ASSISTANT - Phase 1 Demo")
    print("=" * 60)

    # Try to find a real PDF first
    sample_pdf_path = _get_sample_pdf_path()
    if sample_pdf_path:
        print(f"\n  Using real PDF: {os.path.basename(sample_pdf_path)}")
        with open(sample_pdf_path, "rb") as f:
            pdf_bytes = f.read()
    else:
        print("\n  No sample PDF found - using minimal synthetic PDF")
        pdf_bytes = _make_minimal_pdf()

    # Step-by-step pipeline
    _section("Step 1: Loading PDF")
    try:
        import uuid
        doc_id = str(uuid.uuid4())
        page_docs = load_pdf(pdf_bytes, doc_id)
        print(f"  [OK] Document ID  : {doc_id}")
        print(f"  [OK] Pages loaded : {len(page_docs)}")
        for doc in page_docs[:3]:
            preview = doc.page_content[:80].replace("\n", " ")
            print(f"    Page {doc.metadata['page_num']}: {preview}...")
        results["PDF extraction"] = "PASS"
    except Exception as exc:
        print(f"  [FAIL] {exc}")
        results["PDF extraction"] = "FAIL"
        page_docs = []

    _section("Step 2: Extracting Tables")
    try:
        table_docs = extract_tables(pdf_bytes, doc_id)
        print(f"  [OK] Tables found : {len(table_docs)}")
        for t in table_docs:
            print(f"    Page {t.metadata['page_num']}: {t.page_content[:80]}...")
        results["Table extraction"] = "PASS"
    except Exception as exc:
        print(f"  [FAIL] {exc}")
        results["Table extraction"] = "FAIL"
        table_docs = []

    _section("Step 3: Creating Parent/Child Chunks")
    try:
        parent_chunks, child_chunks = create_chunks(page_docs, table_docs)
        print(f"  [OK] Parent chunks : {len(parent_chunks)}")
        print(f"  [OK] Child chunks  : {len(child_chunks)}")

        # Verify table chunks are unsplit
        table_parents = [p for p in parent_chunks if p.metadata["chunk_type"] == "table"]
        print(f"  [OK] Table chunks (unsplit): {len(table_parents)}")

        # Verify parent_id exists on children
        children_with_parent = sum(1 for c in child_chunks if c.metadata.get("parent_id"))
        print(f"  [OK] Children with parent_id: {children_with_parent}/{len(child_chunks)}")

        results["Parent/child chunking"] = "PASS"
        results["Metadata"] = "PASS" if children_with_parent == len(child_chunks) else "FAIL"
    except Exception as exc:
        print(f"  [FAIL] {exc}")
        results["Parent/child chunking"] = "FAIL"
        results["Metadata"] = "FAIL"
        parent_chunks, child_chunks = [], []

    _section("Step 4-7: Building Full Index (embedding model loads here)")
    print("  This may take 15-30 seconds on first run (model download + embedding)...")
    try:
        start = time.time()
        index = build_index(pdf_bytes)
        elapsed = time.time() - start

        print(f"\n  [OK] doc_id          : {index.doc_id}")
        print(f"  [OK] Pages           : {index.num_pages}")
        print(f"  [OK] Tables          : {index.num_tables}")
        print(f"  [OK] Parent chunks   : {len(index.parent_chunks)}")
        print(f"  [OK] Child chunks    : {len(index.child_chunks)}")
        print(f"  [OK] Build time      : {elapsed:.1f}s")

        results["Embeddings"] = "PASS"
        results["Vector index"] = "PASS"
        results["BM25 index"] = "PASS"
    except Exception as exc:
        print(f"  [FAIL] {exc}")
        results["Embeddings"] = "FAIL"
        results["Vector index"] = "FAIL"
        results["BM25 index"] = "FAIL"
        index = None

    _section("Step 8: Ephemeral Storage Check")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    faiss_files = glob.glob(os.path.join(project_root, "**", "*.faiss"), recursive=True)
    if not faiss_files:
        print("  [OK] No FAISS files on disk - index is purely in memory")
        results["Ephemeral storage"] = "PASS"
    else:
        print(f"  [FAIL] FAISS files found: {faiss_files}")
        results["Ephemeral storage"] = "FAIL"

    _section("Step 9: Basic Retrieval Test")
    if index:
        try:
            query = "contract agreement"
            vector_results = index.vector_retriever.invoke(query)
            bm25_results = index.bm25_retriever.invoke(query)
            print(f"  [OK] Vector retriever returned {len(vector_results)} results")
            print(f"  [OK] BM25 retriever returned   {len(bm25_results)} results")

            if vector_results:
                preview = vector_results[0].page_content[:120].replace("\n", " ")
                print(f"\n  Top vector result preview:\n    \"{preview}...\"")

            results["Basic retrieval test"] = "PASS"
        except Exception as exc:
            print(f"  [FAIL] {exc}")
            results["Basic retrieval test"] = "FAIL"
    else:
        results["Basic retrieval test"] = "FAIL"

    # Final Report
    print("\n" + "=" * 60)
    print("  PHASE 1 STATUS")
    print("-" * 60)
    checks = [
        "PDF extraction",
        "Table extraction",
        "Parent/child chunking",
        "Metadata",
        "Embeddings",
        "Vector index",
        "BM25 index",
        "Ephemeral storage",
        "Basic retrieval test",
    ]
    all_passed = True
    for check in checks:
        status = results.get(check, "SKIP")
        icon = "[PASS]" if status == "PASS" else ("[FAIL]" if status == "FAIL" else "[SKIP]")
        if status != "PASS":
            all_passed = False
        print(f"  {icon} {check:<30} {status}")

    print("-" * 60)
    if all_passed:
        print("  All Phase 1 checks PASSED")
    else:
        print("  Some checks failed - see details above")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_demo()
