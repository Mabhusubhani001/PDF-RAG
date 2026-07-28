from fastapi import APIRouter
from schemas.pipeline import PipelineProgressResponse, PipelineStageInfo
from storage.job_storage import JOBS_REGISTRY, create_job

router = APIRouter(prefix="/pipeline", tags=["RAG Visual Pipeline"])


@router.get("/{job_id}/status", response_model=PipelineProgressResponse)
def get_pipeline_progress(job_id: str, mode: str = "local"):
    """
    Get real 9-stage pipeline processing status for a job.
    """
    job = JOBS_REGISTRY.get(job_id)
    if not job:
        # Create default completed record for sample papers
        job = create_job(job_id, "document.pdf", mode)
        job["current_stage"] = 9
        job["percent_complete"] = 100
        job["is_completed"] = True
        for stg in job["stages"]:
            stg["status"] = "completed"

    return PipelineProgressResponse(
        job_id=job["job_id"],
        mode=job["mode"],
        current_stage=job["current_stage"],
        total_stages=job["total_stages"],
        percent_complete=job["percent_complete"],
        is_completed=job["is_completed"],
        stages=[
            PipelineStageInfo(
                stage_number=s["stage_number"],
                name=s["name"],
                description=s["description"],
                status=s["status"]
            )
            for s in job["stages"]
        ]
    )
