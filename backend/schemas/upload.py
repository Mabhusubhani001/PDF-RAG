from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    filename: str
    file_size_bytes: int
    content_type: str
    estimated_pages: int
    estimated_chunks: int


class UploadResponse(BaseModel):
    job_id: str
    filename: str
    status: str
    metadata: DocumentMetadata
    message: str
