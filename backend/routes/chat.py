from typing import Optional
from fastapi import APIRouter, Depends, Header
from schemas.chat import ChatQueryRequest, ChatQueryResponse
from pipelines import LocalRAGPipeline, CloudRAGPipeline
from config import get_settings, Settings

router = APIRouter(prefix="/chat", tags=["RAG Chat & Retrieval"])


@router.post("/query", response_model=ChatQueryResponse)
def execute_rag_query(
    request: ChatQueryRequest,
    settings: Settings = Depends(get_settings),
    x_gemini_api_key: Optional[str] = Header(None),
    x_qdrant_url: Optional[str] = Header(None),
    x_qdrant_api_key: Optional[str] = Header(None)
):
    """
    Execute RAG retrieval and answer synthesis with active document payload filtering.
    """
    if request.mode == "cloud":
        api_key = x_gemini_api_key or settings.GEMINI_API_KEY
        qdrant_url = x_qdrant_url or settings.QDRANT_CLOUD_URL
        qdrant_key = x_qdrant_api_key or settings.QDRANT_CLOUD_API_KEY

        pipeline = CloudRAGPipeline(api_key=api_key, qdrant_url=qdrant_url, qdrant_key=qdrant_key)
    else:
        pipeline = LocalRAGPipeline()

    result = pipeline.query(
        user_question=request.query, 
        top_k=request.top_k, 
        filename=request.filename
    )
    return ChatQueryResponse(**result)
