"""
document_service.py — In-memory document manager for ephemeral session storage.
Reuses existing app.ingestion modules without duplicating code.
"""

import io
from typing import Optional, Tuple
import pypdf

from app.ingestion.index_builder import build_index, DocumentIndex


class DocumentService:
    """
    Singleton-style in-memory service holding the active DocumentIndex.
    No persistence, no SQLite files, purely ephemeral in RAM.
    """

    def __init__(self):
        self.current_index: Optional[DocumentIndex] = None
        self.current_filename: Optional[str] = None
        self.conversation_history: list = []

    def validate_pdf(self, pdf_bytes: bytes) -> Tuple[bool, str]:
        """
        Validate that the uploaded bytes are a valid PDF with extractable text.
        """
        if not pdf_bytes or len(pdf_bytes) == 0:
            return False, "The uploaded file is empty. Please upload a valid PDF file."

        if not pdf_bytes.startswith(b"%PDF"):
            return False, "Invalid file format: file header is missing %PDF."

        try:
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            total_text_len = sum(len(p.extract_text() or "") for p in reader.pages)
            if total_text_len < 10:
                return False, "This PDF does not contain extractable text. Scanned/image-only PDFs are not supported."
        except Exception as exc:
            return False, f"Could not parse the uploaded PDF: {exc}"

        return True, ""

    def ingest_document(self, filename: str, pdf_bytes: bytes) -> DocumentIndex:
        """
        Builds the in-memory Chroma + BM25 index and resets conversation state.
        Discards any previous document index.
        """
        is_valid, err_msg = self.validate_pdf(pdf_bytes)
        if not is_valid:
            raise ValueError(err_msg)

        # Build ephemeral index using existing Phase 1 & 2 pipeline
        index = build_index(pdf_bytes)

        # Store only the current document state
        self.current_index = index
        self.current_filename = filename
        self.conversation_history = []

        return index

    def clear(self):
        """Explicitly reset state."""
        self.current_index = None
        self.current_filename = None
        self.conversation_history = []

    def get_status(self) -> dict:
        if self.current_index:
            return {
                "document_loaded": True,
                "filename": self.current_filename,
                "pages": self.current_index.num_pages,
                "chunks": len(self.current_index.child_chunks),
            }
        return {
            "document_loaded": False,
            "filename": None,
            "pages": 0,
            "chunks": 0,
        }


# Global instance for the running FastAPI server
document_service = DocumentService()
