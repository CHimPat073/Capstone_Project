"""
test_phase4.py — Phase 4 Verification: UI Guardrails, Ephemeral Lifecycle, & Evaluation Metrics.

Runs automatically with:   pytest tests/test_phase4.py -v
Or directly with:          python tests/test_phase4.py
"""

import glob
import os
import sys
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestion.index_builder import build_index
from app.reasoning.agent_graph import answer
from app.ui.streamlit_app import validate_pdf_bytes
from app.evaluation.ragas_evaluator import (
    calculate_faithfulness,
    calculate_context_precision,
    calculate_context_recall,
    calculate_answer_relevancy,
    is_correct_refusal,
)


def _make_pdf(text: str) -> bytes:
    """Helper to generate minimal valid single-page PDF with custom text."""
    pdf_template = f"""%PDF-1.4
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
<< /Length {len(text) + 30} >>
stream
BT /F1 12 Tf 50 700 Td ({text}) Tj ET
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
0000000400 00000 n 

trailer
<< /Size 6 /Root 1 0 R >>
startxref
480
%%EOF"""
    return pdf_template.encode("latin1")


class TestPhase4GuardrailsAndLifecycle:
    """Tests for Phase 4 guardrails, ephemeral isolation, and metrics."""

    def test_pdf_validation_guardrails(self):
        """Validate rejection of non-PDF, empty bytes, and scanned PDFs."""
        # Non-PDF
        valid, msg = validate_pdf_bytes(b"Plain text file not a PDF")
        assert not valid
        assert "PDF" in msg

        # Empty bytes
        valid, msg = validate_pdf_bytes(b"")
        assert not valid
        assert "empty" in msg.lower()

        # Scanned / empty page PDF (valid PDF header but no extractable text)
        empty_page_pdf = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >> endobj
xref
0 4
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
trailer << /Size 4 /Root 1 0 R >>
startxref
190
%%EOF"""
        valid, msg = validate_pdf_bytes(empty_page_pdf)
        assert not valid
        assert "scanned" in msg.lower() or "extractable text" in msg.lower()

    def test_ephemeral_lifecycle_isolation(self):
        """Confirm document A index does NOT leak into document B index."""
        pdf_a = _make_pdf("Contract A Confidential Secret Code Alpha 999")
        pdf_b = _make_pdf("Contract B Unrelated Terms Bravo 888")

        index_a = build_index(pdf_a)
        res_a = index_a.vector_retriever.invoke("Alpha 999")
        assert any("Alpha 999" in d.page_content for d in res_a)

        # Discard index A cleanly
        del index_a

        # Build index B
        index_b = build_index(pdf_b)
        res_b = index_b.vector_retriever.invoke("Alpha 999")
        # Ensure Contract A content is NOT found in Index B
        assert not any("Alpha 999" in d.page_content for d in res_b)
        assert any("Bravo 888" in d.page_content for d in index_b.vector_retriever.invoke("Bravo 888"))

    def test_out_of_domain_refusal(self):
        """Confirm out-of-domain questions trigger low confidence refusal."""
        pdf_bytes = _make_pdf("Agreement regarding software development payment terms of 5000 USD.")
        index = build_index(pdf_bytes)

        result = answer(index, "Who won the 2022 FIFA World Cup in Qatar?")
        assert result["confidence"] == "low"
        assert is_correct_refusal(result["answer"], result["confidence"], is_out_of_domain=True)

    def test_ragas_metrics_calculations(self):
        """Verify Ragas metric calculation functions return expected bounds."""
        from langchain_core.documents import Document
        docs = [
            Document(page_content="Termination requires 30 days written notice to the other party.", metadata={"page_num": 1}),
            Document(page_content="Governing law is New York State.", metadata={"page_num": 2}),
        ]

        # Faithfulness: 1.0 for verified claim
        faith = calculate_faithfulness("Termination requires 30 days notice [Page 1].", docs, is_grounded=True, unsupported_claims=[])
        assert faith == 1.0

        # Context Precision: top chunk has relevant termination terms
        prec = calculate_context_precision("Termination requires 30 days notice", docs, k=2)
        assert prec >= 0.5

        # Context Recall: relevant terms retrieved
        recall = calculate_context_recall("Termination requires 30 days notice", docs)
        assert recall >= 0.7

        # Answer Relevancy
        rel = calculate_answer_relevancy("What is the termination notice period?", "Termination requires 30 days notice.")
        assert 0.0 <= rel <= 1.0

    def test_streamlit_app_imports_cleanly(self):
        """Confirm Streamlit UI module imports without error."""
        import app.ui.streamlit_app as ui_app
        assert hasattr(ui_app, "main")
        assert hasattr(ui_app, "validate_pdf_bytes")


def run_demo():
    print("\n" + "=" * 60)
    print("  PHASE 4 VERIFICATION SUITE")
    print("=" * 60)
    t = TestPhase4GuardrailsAndLifecycle()
    t.test_pdf_validation_guardrails()
    print("  [PASS] PDF Validation Guardrails")
    t.test_ephemeral_lifecycle_isolation()
    print("  [PASS] Ephemeral Lifecycle Isolation")
    t.test_out_of_domain_refusal()
    print("  [PASS] Out-of-Domain Refusal")
    t.test_ragas_metrics_calculations()
    print("  [PASS] Ragas Metric Calculations")
    t.test_streamlit_app_imports_cleanly()
    print("  [PASS] Streamlit App Module Check")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_demo()
