import uuid
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Header
from schemas.upload import UploadResponse, DocumentMetadata
from config import get_settings, Settings
from pipelines.local_pipeline import LocalRAGPipeline
from pipelines.cloud_pipeline import CloudRAGPipeline

router = APIRouter(prefix="/upload", tags=["PDF Document Upload"])


@router.post("", response_model=UploadResponse)
async def upload_pdf_document(
    file: UploadFile = File(...),
    mode: str = Form("local"),
    settings: Settings = Depends(get_settings),
    x_gemini_api_key: Optional[str] = Header(None),
    x_qdrant_url: Optional[str] = Header(None),
    x_qdrant_api_key: Optional[str] = Header(None)
):
    """
    Ingest, parse, chunk, embed, and index PDF document via Local or Cloud RAG pipeline.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    file_bytes = await file.read()
    job_id = f"job_{uuid.uuid4().hex[:8]}"

    try:
        if mode == "cloud":
            api_key = x_gemini_api_key or settings.GEMINI_API_KEY
            qdrant_url = x_qdrant_url or settings.QDRANT_CLOUD_URL
            qdrant_key = x_qdrant_api_key or settings.QDRANT_CLOUD_API_KEY

            pipeline = CloudRAGPipeline(api_key=api_key, qdrant_url=qdrant_url, qdrant_key=qdrant_key)
            result = pipeline.ingest_document(file_bytes, file.filename, job_id=job_id)
        else:
            pipeline = LocalRAGPipeline()
            result = pipeline.ingest_document(file_bytes, file.filename, job_id=job_id)

        estimated_pages = result.get("total_pages", 1)
        estimated_chunks = result.get("total_chunks", 2)

        return UploadResponse(
            job_id=job_id,
            filename=file.filename,
            status="completed",
            metadata=DocumentMetadata(
                filename=file.filename,
                file_size_bytes=len(file_bytes),
                content_type=file.content_type or "application/pdf",
                estimated_pages=estimated_pages,
                estimated_chunks=estimated_chunks
            ),
            message=f"Document '{file.filename}' processed via {mode.upper()} RAG pipeline. Created {estimated_chunks} vector chunks."
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
