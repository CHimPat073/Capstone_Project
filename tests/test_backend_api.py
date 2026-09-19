"""
test_api.py — Verification of FastAPI endpoints and backend services.
"""

import io
import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.main import app
from backend.services.document_service import document_service

client = TestClient(app)


def test_health_endpoint():
    """1. GET /api/health should return status ok"""
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_status_no_document():
    """2. GET /api/status when no document is loaded"""
    document_service.clear()
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["document_loaded"] is False
    assert data["filename"] is None


def test_chat_without_document():
    """3. POST /api/chat when no document is loaded should return refusal"""
    document_service.clear()
    res = client.post("/api/chat", json={"question": "What is the payment term?"})
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is False
    assert "No document is currently loaded" in data["answer"]


def test_upload_invalid_file_extension():
    """4. POST /api/upload with non-PDF extension should return 400"""
    files = {"file": ("test.txt", b"plain text", "text/plain")}
    res = client.post("/api/upload", files=files)
    assert res.status_code == 400


def test_upload_valid_pdf_and_query():
    """5. Full roundtrip: Upload real contract PDF -> check status -> ask question"""
    pdf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "data", "cuad_contract_0.pdf")
    if not os.path.exists(pdf_path):
        pytest.skip("Test contract cuad_contract_0.pdf not found")

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    files = {"file": ("cuad_contract_0.pdf", pdf_bytes, "application/pdf")}
    res = client.post("/api/upload", files=files)
    assert res.status_code == 200
    up_data = res.json()
    assert up_data["success"] is True
    assert up_data["pages"] >= 1
    assert up_data["chunks"] >= 1

    # Check status endpoint
    status_res = client.get("/api/status")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["document_loaded"] is True
    assert status_data["filename"] == "cuad_contract_0.pdf"

    # Test in-domain chat
    chat_res = client.post("/api/chat", json={"question": "What are the payment terms?"})
    assert chat_res.status_code == 200
    chat_data = chat_res.json()
    assert chat_data["success"] is True
    assert chat_data["status"] == "answered"
    assert len(chat_data["citations"]) > 0
    assert len(chat_data["sources"]) > 0
    assert chat_data["loop"]["grounding_passed"] is True
    # Test evaluation and document relevance metrics
    assert chat_data["eval_metrics"] is not None
    assert chat_data["eval_metrics"]["document_relevance_score"] >= 0.4
    assert chat_data["eval_metrics"]["relevance_level"] in ["High", "Moderate"]
    assert "context_alignment" in chat_data["eval_metrics"]
    assert "faithfulness" in chat_data["eval_metrics"]
    assert "answer_relevancy" in chat_data["eval_metrics"]

    # Test out-of-domain chat
    ood_res = client.post("/api/chat", json={"question": "What is the capital of France?"})
    assert ood_res.status_code == 200
    ood_data = ood_res.json()
    assert ood_data["success"] is False
    assert ood_data["status"] == "out_of_domain"
    assert "outside the scope" in ood_data["answer"].lower()
    assert ood_data["eval_metrics"] is not None
    assert ood_data["eval_metrics"]["document_relevance_score"] <= 0.2
    assert "Low" in ood_data["eval_metrics"]["relevance_level"]

    # Test vague/clarification chat
    vague_res = client.post("/api/chat", json={"question": "what"})
    assert vague_res.status_code == 200
    vague_data = vague_res.json()
    assert vague_data["success"] is False
    assert vague_data["status"] == "needs_clarification"
    assert len(vague_data["suggestions"]) > 0
