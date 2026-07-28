from typing import List, Optional
from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    chunk_id: str
    page_number: int
    similarity_score: float
    text_snippet: str
    filename: Optional[str] = None
    metadata: Optional[dict] = None


class ChatQueryRequest(BaseModel):
    query: str
    mode: str = "local"  # "local" or "cloud"
    top_k: int = 3
    filename: Optional[str] = None
    collection_name: Optional[str] = None


class ChatQueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    latency_ms: int
    llm_model: str
    embedding_model: str
    collection_name: str
