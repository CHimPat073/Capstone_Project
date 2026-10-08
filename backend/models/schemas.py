"""
schemas.py — Pydantic models for the FastAPI backend.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class DocumentUploadResponse(BaseModel):
    success: bool
    filename: str
    message: str
    pages: int
    chunks: int


class DocumentStatus(BaseModel):
    document_loaded: bool
    filename: Optional[str] = None
    pages: int = 0
    chunks: int = 0


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural language user question")


class Citation(BaseModel):
    page: int
    text: Optional[str] = None
    bbox: Optional[List[float]] = None
    page_width: Optional[float] = None
    page_height: Optional[float] = None


class Source(BaseModel):
    page: int
    text: str
    chunk_type: Optional[str] = "text"


class LoopStatus(BaseModel):
    query_relevant: bool = True
    retrieval_attempts: int = 1
    rewrites: int = 0
    grounding_passed: bool = True
    original_query: Optional[str] = None
    resolved_query: Optional[str] = None


class EvalMetrics(BaseModel):
    document_relevance_score: float = Field(..., description="Relevance score to document context (0.0 to 1.0)")
    relevance_level: str = Field(..., description="High, Moderate, or Low")
    context_alignment: float = Field(..., description="Overlap / similarity between query and retrieved document context")
    faithfulness: float = Field(..., description="Factual grounding of answer in retrieved chunks")
    answer_relevancy: float = Field(..., description="Relevancy of answer to the user prompt")


class ChatResponse(BaseModel):
    success: bool
    answer: str
    status: str  # "answered", "needs_clarification", "out_of_domain", "retrieval_failed", "grounding_failed"
    citations: List[Citation] = []
    sources: List[Source] = []
    loop: LoopStatus
    eval_metrics: Optional[EvalMetrics] = None
    suggestions: List[str] = []
    evaluation_id: Optional[str] = None


class HumanFeedback(BaseModel):
    evaluation_id: str = Field(..., min_length=1)
    rating: int = Field(..., ge=1, le=5)
    helpful: Optional[bool] = None
    comment: Optional[str] = Field(default=None, max_length=2000)
