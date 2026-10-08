"""
routes.py — FastAPI route handlers for the Document Intelligence Assistant.
"""

import io

import pypdf
import pypdfium2 as pdfium
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response
from backend.models.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentStatus,
    DocumentUploadResponse,
    HealthResponse,
    HumanFeedback,
)
from backend.services.chat_service import chat_service
from backend.services.document_service import document_service

router = APIRouter(prefix="/api")
human_feedback = []


@router.get("/health", response_model=HealthResponse)
def health():
    """1. GET /api/health"""
    return HealthResponse(status="ok")


@router.get("/status", response_model=DocumentStatus)
def get_status():
    """3. GET /api/status"""
    return DocumentStatus(**document_service.get_status())


@router.get("/document/page/{page_number}")
def get_document_page(page_number: int):
    """Render a single PDF page to PNG in memory for the document canvas."""
    content = document_service.current_pdf_bytes
    if not content:
        raise HTTPException(status_code=404, detail="No PDF is loaded.")
    try:
        document = pdfium.PdfDocument(content)
        if page_number < 1 or page_number > len(document):
            raise HTTPException(status_code=404, detail="Page does not exist.")
        page = document[page_number - 1]
        bitmap = page.render(scale=1.5)
        image = bitmap.to_pil()
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        page.close()
        document.close()
        return Response(content=buffer.getvalue(), media_type="image/png",
                        headers={"Cache-Control": "no-store"})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not render PDF page: {exc}") from exc


@router.get("/document/outline")
def get_document_outline():
    """Return PDF bookmark entries when the source file contains them."""
    content = document_service.current_pdf_bytes
    if not content:
        raise HTTPException(status_code=404, detail="No PDF is loaded.")
    reader = pypdf.PdfReader(io.BytesIO(content))
    entries = []

    def visit(items, depth=0):
        for item in items:
            if isinstance(item, list):
                visit(item, depth + 1)
                continue
            try:
                page = reader.get_destination_page_number(item) + 1
                entries.append({"title": str(item.title), "page": page, "depth": depth})
            except Exception:
                continue

    visit(reader.outline)
    return {"entries": entries}


@router.get("/document/text/{page_number}")
def get_document_page_text(page_number: int):
    """Return extracted page text for the viewer's lightweight find-in-page."""
    content = document_service.current_pdf_bytes
    if not content:
        raise HTTPException(status_code=404, detail="No PDF is loaded.")
    reader = pypdf.PdfReader(io.BytesIO(content))
    if page_number < 1 or page_number > len(reader.pages):
        raise HTTPException(status_code=404, detail="Page does not exist.")
    return {"page": page_number, "text": reader.pages[page_number - 1].extract_text() or ""}


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    2. POST /api/upload
    Accepts PDF via multipart/form-data, builds ephemeral index in RAM,
    resets previous document and chat session.
    """
    if not (file.filename or "").lower().endswith(".pdf"):
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
        response = chat_service.process_chat(request.question, retrieval_mode=request.retrieval_mode)
        return response
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during chat processing: {str(exc)}",
        )


@router.delete("/chat/history")
def clear_chat_history():
    """Clear the current in-memory conversation while keeping the active PDF."""
    document_service.conversation_history = []
    return {"cleared": True}


@router.post("/evaluation/feedback")
def submit_human_feedback(feedback: HumanFeedback):
    """Collect explicit answer ratings in process memory for human evaluation."""
    human_feedback.append(feedback.dict())
    return {"accepted": True, "count": len(human_feedback)}


@router.get("/evaluation/summary")
def human_evaluation_summary():
    """Return aggregate feedback without retaining uploaded document content."""
    ratings = [row["rating"] for row in human_feedback]
    return {
        "responses": len(ratings),
        "average_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
        "helpful_rate": round(sum(row.get("helpful") is True for row in human_feedback) / len(ratings), 3) if ratings else None,
    }
