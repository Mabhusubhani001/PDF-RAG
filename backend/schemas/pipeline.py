from typing import List, Optional
from pydantic import BaseModel


class PipelineStageInfo(BaseModel):
    stage_number: int
    name: str
    description: str
    status: str  # "queued", "in_progress", "completed", "error"
    execution_time_ms: Optional[int] = None


class PipelineProgressResponse(BaseModel):
    job_id: str
    mode: str
    current_stage: int
    total_stages: int = 9
    percent_complete: int
    is_completed: bool
    stages: List[PipelineStageInfo]
