"""Regression coverage for document chat failures found through browser testing."""

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.reasoning.agent_graph import (
    _bbox_for_text,
    _is_full_document_extraction_query,
    answer,
)
from app.reasoning.llm_client import LLMClient
from backend.main import app
from backend.models.schemas import ChatResponse, LoopStatus


def _document(text: str, page: int) -> Document:
    return Document(page_content=text, metadata={"page_num": page, "chunk_type": "text"})


def test_short_human_name_request_uses_full_document_extraction():
    assert _is_full_document_extraction_query("only human name") is True
    pages = [
        _document("Diksha Damahe worked under Mr. Abhinav Arvind at Hindustan Copper Limited.", 1),
        _document("The report was approved by Dr. Pradeep Kumar Mishra.", 2),
    ]
    index = SimpleNamespace(page_docs=pages, child_chunks=pages)

    result = answer(index, "only human name", llm_client=LLMClient())

    assert result["route_strategy"] == "full_document_extraction"
    assert "Diksha Damahe" in result["answer"]
    assert "Mr. Abhinav Arvind" in result["answer"]
    assert "Dr. Pradeep Kumar Mishra" in result["answer"]
    assert "Hindustan Copper Limited" not in result["answer"]


def test_summary_does_not_prioritize_cover_page_boilerplate():
    pages = [
        _document("INTERNSHIP REPORT submitted for the Award of Degree and certificate requirements.", 1),
        _document("The project implemented an anomaly detection platform for industrial equipment.", 2),
        _document("Testing verified 92 percent precision and latency below four seconds.", 3),
    ]
    index = SimpleNamespace(page_docs=pages, child_chunks=pages)

    result = answer(index, "Summarize the document", llm_client=LLMClient())

    assert result["confidence"] == "high"
    assert "anomaly detection platform" in result["answer"]
    assert "92 percent precision" in result["answer"]


def test_comparison_fallback_keeps_both_sides_separate():
    client = LLMClient()
    prompt = """Question: Compare project objectives and reported results

Context:
[Page 3]
The project objectives were to digitize shift records and forecast copper recovery.

[Page 8]
The reported results included an R2 score of 0.972 and latency below four seconds.
"""
    result = client.call_json("You are an expert Document Intelligence Assistant.", prompt)

    assert "project objectives:" in result["answer"].lower()
    assert "reported results:" in result["answer"].lower()
    assert {citation["page"] for citation in result["citations"]} == {3, 8}


def test_semantic_intent_paraphrase_grades_objectives_context():
    client = LLMClient()
    result = client.call_json(
        "You are a context evaluation grader.",
        "Question: What was the project intended to achieve?\n\n"
        "Context:\n[Page 7]\nLearning objectives and project goals included building a recovery forecasting system.",
    )
    assert result["sufficient"] is True


def test_precise_bbox_uses_only_cited_words():
    words = [
        {"text": "INTERNSHIP", "x0": 50, "top": 70, "x1": 120, "bottom": 82},
        {"text": "REPORT", "x0": 125, "top": 70, "x1": 175, "bottom": 82},
        {"text": "Submitted", "x0": 50, "top": 95, "x1": 110, "bottom": 107},
    ]
    page = Document(page_content="INTERNSHIP REPORT Submitted", metadata={
        "page_num": 1,
        "page_width": 612,
        "page_height": 792,
        "bbox_words_json": json.dumps(words),
    })

    geometry = _bbox_for_text(SimpleNamespace(page_docs=[page]), 1, "INTERNSHIP REPORT")

    assert geometry["bbox"] == [50.0, 70.0, 175.0, 82.0]


def test_chat_response_lists_are_not_shared():
    first = ChatResponse(success=True, answer="one", status="answered", loop=LoopStatus())
    second = ChatResponse(success=True, answer="two", status="answered", loop=LoopStatus())
    first.suggestions.append("next")
    assert second.suggestions == []


def test_backend_rejects_pdf_larger_than_20_mb():
    client = TestClient(app)
    oversized = b"%PDF" + b"0" * (20 * 1024 * 1024)
    response = client.post("/api/upload", files={"file": ("large.pdf", oversized, "application/pdf")})
    assert response.status_code == 413


def test_date_citation_bbox_matches_slashed_date():
    words = [
        {"text": "Contract", "x0": 10.0, "top": 20.0, "x1": 60.0, "bottom": 32.0},
        {"text": "Date:", "x0": 65.0, "top": 20.0, "x1": 95.0, "bottom": 32.0},
        {"text": "12/9/2019", "x0": 100.0, "top": 20.0, "x1": 155.0, "bottom": 32.0},
    ]
    page = Document(page_content="Contract Date: 12/9/2019", metadata={
        "page_num": 1,
        "page_width": 612,
        "page_height": 792,
        "bbox_words_json": json.dumps(words),
    })
    geometry = _bbox_for_text(SimpleNamespace(page_docs=[page]), 1, "12/9/2019")
    assert geometry is not None
    assert geometry["bbox"] == [100.0, 20.0, 155.0, 32.0]


def test_human_name_filtering_rejects_legal_and_trade_terms():
    pages = [
        _document("Quarantine Bureau inspects All Risks and War Risk under Force Majeure.", 1),
        _document("Signed by authorized representative Diksha Damahe.", 2),
    ]
    index = SimpleNamespace(page_docs=pages, child_chunks=pages)
    result = answer(index, "only human name", llm_client=LLMClient())
    assert "Diksha Damahe" in result["answer"]
    assert "Quarantine Bureau" not in result["answer"]
    assert "Force Majeure" not in result["answer"]
    assert "War Risk" not in result["answer"]


def test_morphological_stem_grading_matches_payment_variation():
    client = LLMClient()
    result = client.call_json(
        "You are a context evaluation grader.",
        "Question: What is the process for paying for products?\n\n"
        "Context:\n[Page 2]\nThe Buyer shall make payment through an irrevocable letter of credit for products.",
    )
    assert result["sufficient"] is True


def test_unrelated_query_classified_as_out_of_domain():
    from backend.services.chat_service import chat_service
    from backend.services.document_service import document_service

    pages = [_document("This agreement governs the delivery of copper cathodes.", 1)]
    document_service.current_index = SimpleNamespace(
        page_docs=pages,
        child_chunks=pages,
        num_pages=1,
    )
    response = chat_service.process_chat("What is the recipe for chocolate cake?")
    assert response.status == "out_of_domain"
    assert response.success is False

