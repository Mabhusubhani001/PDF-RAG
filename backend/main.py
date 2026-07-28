from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import get_settings
from routes import health_router, upload_router, pipeline_router, chat_router

settings = get_settings()

app = FastAPI(
    title="PDF RAG Playground API Engine",
    description="Modular FastAPI backend powering Local (Ollama + Qdrant) and Cloud (Gemini + Qdrant Cloud) PDF RAG Pipelines.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for Astro frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(health_router, prefix="/api/v1")
app.include_router(upload_router, prefix="/api/v1")
app.include_router(pipeline_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")


@app.get("/")
def read_root():
    return {
        "name": "PDF RAG Playground Backend Engine",
        "version": "1.0.0",
        "status": "online",
        "docs": "/docs",
        "mode_configurations": {
            "local": {
                "ollama_url": settings.OLLAMA_BASE_URL,
                "qdrant_url": settings.LOCAL_QDRANT_URL,
                "llm": settings.LOCAL_LLM_MODEL,
                "embedding": settings.LOCAL_EMBEDDING_MODEL,
                "collection": settings.LOCAL_COLLECTION_NAME
            },
            "cloud": {
                "qdrant_cloud": settings.QDRANT_CLOUD_URL or "Unconfigured",
                "llm": settings.CLOUD_LLM_MODEL,
                "embedding": settings.CLOUD_EMBEDDING_MODEL,
                "collection": settings.CLOUD_COLLECTION_NAME
            }
        }
    }
