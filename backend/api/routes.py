"""
routes.py — FastAPI route handlers for the Document Intelligence Assistant.
"""

from fastapi import APIRouter, File, HTTPException, UploadFile
from backend.models.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentStatus,
    DocumentUploadResponse,
    HealthResponse,
)
from backend.services.chat_service import chat_service
from backend.services.document_service import document_service

router = APIRouter(prefix="/api")


@router.get("/health", response_model=HealthResponse)
def health():
    """1. GET /api/health"""
    return HealthResponse(status="ok")


@router.get("/status", response_model=DocumentStatus)
def get_status():
    """3. GET /api/status"""
    return DocumentStatus(**document_service.get_status())


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    2. POST /api/upload
    Accepts PDF via multipart/form-data, builds ephemeral index in RAM,
    resets previous document and chat session.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only PDF files (.pdf) are supported.",
        )

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")

        index = document_service.ingest_document(file.filename, content)

        return DocumentUploadResponse(
            success=True,
            filename=file.filename,
            message="Document processed successfully",
            pages=index.num_pages,
            chunks=len(index.child_chunks),
        )
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the document: {str(exc)}",
        )


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    4. POST /api/chat
    Executes the grounded reasoning loop and returns safe workflow metadata.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        response = chat_service.process_chat(request.question)
        return response
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during chat processing: {str(exc)}",
        )
