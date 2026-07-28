from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from schemas.health import HealthCheckResponse, ServiceStatus
from config import get_settings, Settings
from services.llm_service import LLMService
from services.vector_store import VectorStoreService

router = APIRouter(prefix="/health", tags=["Health & Diagnostics"])


@router.get("", response_model=HealthCheckResponse)
def get_system_health(settings: Settings = Depends(get_settings)):
    """
    Real-time diagnostic check pinging local Ollama, local Qdrant, and Qdrant Cloud.
    """
    # 1. Ping Ollama
    ollama_service = LLMService(provider="local", base_url=settings.OLLAMA_BASE_URL)
    ollama_info = ollama_service.check_ollama_health()
    ollama_running = (ollama_info.get("status") == "running")
    installed_models = ollama_info.get("models", [])
    
    has_llama = any("llama" in m.lower() for m in installed_models)
    has_embed = any("embed" in m.lower() or "qwen" in m.lower() for m in installed_models)
    models_ready = ollama_running and (has_llama or has_embed or len(installed_models) > 0)

    # 2. Ping Qdrant Local
    qdrant_service = VectorStoreService(qdrant_url=settings.LOCAL_QDRANT_URL)
    qdrant_info = qdrant_service.check_health()
    qdrant_running = (qdrant_info.get("status") == "running")

    # Overall local readiness
    overall_ready = ollama_running and qdrant_running

    return HealthCheckResponse(
        mode="all",
        ready=overall_ready,
        services={
            "ollama": ServiceStatus(
                name="Ollama Engine",
                status="running" if ollama_running else "error",
                endpoint=settings.OLLAMA_BASE_URL,
                details=f"Port 11434 {'Active' if ollama_running else 'Offline (Run: ollama serve)'}"
            ),
            "local_qdrant": ServiceStatus(
                name="Qdrant Local (Docker)",
                status="running" if qdrant_running else "error",
                endpoint=settings.LOCAL_QDRANT_URL,
                details=f"Port 6333 {'Active' if qdrant_running else 'Offline (Run Qdrant container)'}"
            ),
            "models": ServiceStatus(
                name="Required Models",
                status="running" if models_ready else ("pending" if ollama_running else "error"),
                endpoint=settings.LOCAL_LLM_MODEL,
                details=f"Models: {', '.join(installed_models) if installed_models else 'None installed (ollama pull llama3.1)'}"
            ),
            "cloud_qdrant": ServiceStatus(
                name="Qdrant Cloud Cluster",
                status="running" if settings.QDRANT_CLOUD_URL else "pending",
                endpoint=settings.QDRANT_CLOUD_URL or "Not Configured",
                details=f"Collection: {settings.CLOUD_COLLECTION_NAME}"
            )
        },
        timestamp=datetime.now(timezone.utc).isoformat()
    )
