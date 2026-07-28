from typing import Dict, Any, List

# In-memory registry storing real pipeline job progress
JOBS_REGISTRY: Dict[str, Dict[str, Any]] = {}


def create_job(job_id: str, filename: str, mode: str) -> Dict[str, Any]:
    job_data = {
        "job_id": job_id,
        "filename": filename,
        "mode": mode,
        "current_stage": 1,
        "total_stages": 9,
        "percent_complete": 11,
        "is_completed": False,
        "error": None,
        "metadata": {},
        "stages": [
            {"stage_number": 1, "name": "Reading PDF Structure", "description": "Opening file stream & validating header", "status": "in_progress"},
            {"stage_number": 2, "name": "Extracting Pages", "description": "Extracting text page-by-page", "status": "queued"},
            {"stage_number": 3, "name": "Splitting Document Text", "description": "Normalizing paragraphs & line breaks", "status": "queued"},
            {"stage_number": 4, "name": "Creating Overlapping Chunks", "description": "Building 500-char sliding windows", "status": "queued"},
            {"stage_number": 5, "name": "Generating Vector Embeddings", "description": "Embedding dense vectors via AI model", "status": "queued"},
            {"stage_number": 6, "name": "Connecting to Vector DB", "description": "Verifying Qdrant collection", "status": "queued"},
            {"stage_number": 7, "name": "Storing Vectors & Payload", "description": "Upserting points to Qdrant index", "status": "queued"},
            {"stage_number": 8, "name": "Finalizing HNSW Graph Index", "description": "Building spatial graph index", "status": "queued"},
            {"stage_number": 9, "name": "Ready for Question Answering", "description": "Vector collection online", "status": "queued"}
        ]
    }
    JOBS_REGISTRY[job_id] = job_data
    return job_data


def update_job_stage(job_id: str, stage_number: int, stage_desc: str = "", metadata_update: Dict[str, Any] = None):
    if job_id not in JOBS_REGISTRY:
        return

    job = JOBS_REGISTRY[job_id]
    job["current_stage"] = stage_number
    job["percent_complete"] = Math_round((stage_number / 9) * 100)

    if metadata_update:
        job["metadata"].update(metadata_update)

    for stg in job["stages"]:
        sn = stg["stage_number"]
        if sn < stage_number:
            stg["status"] = "completed"
        elif sn == stage_number:
            stg["status"] = "in_progress"
            if stage_desc:
                stg["description"] = stage_desc
        else:
            stg["status"] = "queued"

    if stage_number >= 9:
        job["is_completed"] = True
        job["percent_complete"] = 100
        for stg in job["stages"]:
            stg["status"] = "completed"


def set_job_error(job_id: str, error_msg: str):
    if job_id in JOBS_REGISTRY:
        job = JOBS_REGISTRY[job_id]
        job["error"] = error_msg
        for stg in job["stages"]:
            if stg["status"] == "in_progress":
                stg["status"] = "error"
                stg["description"] = f"Error: {error_msg}"


def Math_round(val: float) -> int:
    return int(val + 0.5)
