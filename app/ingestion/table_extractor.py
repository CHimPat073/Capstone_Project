"""
table_extractor.py — Phase 1: Detect tables in a PDF and return them as Documents.

WHY we use camelot: It finds structured table boundaries in PDFs, which pure
text extraction misses. Each table becomes ONE Document (never split) so that
row/column relationships are preserved during retrieval.

WHY we convert to Markdown: Markdown tables are readable plain text that the
embedding model can handle. They also make it obvious during debugging that a
chunk is a table.

NOTE: camelot requires the PDF as a file path (not bytes). We write the bytes
to a NamedTemporaryFile, extract tables, then delete it immediately. This is
the one unavoidable disk touch — the main PDF store remains in-memory.
"""

import os
import tempfile
from typing import List

from langchain_core.documents import Document


def extract_tables(pdf_bytes: bytes, doc_id: str) -> List[Document]:
    """
    Use camelot to detect tables in the PDF and return one Document per table.

    Args:
        pdf_bytes: Raw bytes of the PDF.
        doc_id:    The same doc_id used by load_pdf for this PDF.

    Returns:
        A list of Documents (possibly empty if no tables detected), each with:
            doc_id     – same as the parent PDF
            page_num   – page where the table was found
            chunk_type – "table"
            parent_id  – empty string (tables are never split, so no parent)
            token_count– approximate
    """

    try:
        import camelot  # lazy import — camelot is optional; graceful fallback below
    except ImportError:
        # camelot is not installed; table extraction is skipped silently.
        # The pipeline continues with text-only chunks.
        print("[table_extractor] camelot not available — skipping table extraction.")
        return []

    table_docs: List[Document] = []

    # camelot needs a real file path, so we write the bytes to a temp file.
    # delete=False lets us read it, then we remove it ourselves.
    tmp_path = None
    try:
        # mkstemp gives us an OS-level file descriptor (fd) and a path.
        # We write the bytes, then EXPLICITLY close the fd before calling camelot.
        # On Windows, NamedTemporaryFile holds a lock even after the with-block;
        # mkstemp + manual close avoids that lock entirely.
        fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        try:
            os.write(fd, pdf_bytes)
        finally:
            os.close(fd)  # release the file handle so camelot can open it

        # "lattice" mode uses table borders (good for structured contracts).
        # "stream" mode uses whitespace gaps (good for borderless tables).
        # We try lattice first; stream is available for Phase 2 experimentation.
        try:
            tables = camelot.read_pdf(tmp_path, pages="all", flavor="lattice")
        except Exception as exc:
            # Some PDFs crash camelot (encrypted, malformed). Skip gracefully.
            print(f"[table_extractor] camelot failed: {exc} - skipping tables.")
            return []

        for table in tables:
            page_num = table.page  # camelot gives us the 1-based page number

            # Convert the pandas DataFrame to Markdown text
            try:
                markdown_text = table.df.to_markdown(index=False)
            except Exception:
                # to_markdown needs the 'tabulate' package; fall back to CSV-style
                markdown_text = table.df.to_csv(index=False)

            if not markdown_text or not markdown_text.strip():
                continue  # skip empty tables

            approx_tokens = len(markdown_text) // 4

            doc = Document(
                page_content=markdown_text,
                metadata={
                    "doc_id": doc_id,
                    "page_num": page_num,
                    "chunk_type": "table",
                    "parent_id": "",       # tables are never split into children
                    "token_count": approx_tokens,
                },
            )
            table_docs.append(doc)

    finally:
        # Try to clean up the temp file.
        # On Windows, camelot (via pdfminer) may keep a handle open briefly,
        # causing WinError 32. We swallow that — the OS temp dir cleans itself.
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass  # Windows file lock — temp file will be cleaned by OS

    return table_docs
