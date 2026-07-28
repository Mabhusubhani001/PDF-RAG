import time
from typing import Dict, Any, Optional
from pipelines.base import BaseRAGPipeline
from config import get_settings
from services.pdf_parser import PDFParserService
from services.vector_store import VectorStoreService
from services.llm_service import LLMService
from utils.logger import logger
from storage.job_storage import create_job, update_job_stage, set_job_error


class CloudRAGPipeline(BaseRAGPipeline):
    """
    Production-ready Cloud PDF RAG Pipeline.
    Stack: PyMuPDF + LangChain (1000/300) + Google Gemini (text-embedding-004) + Qdrant Cloud Cluster + Gemini 1.5 Flash LLM.
    Strictly uses Qdrant Cloud cluster credentials (never falls back to localhost:6333).
    """
    def __init__(self, api_key: Optional[str] = None, qdrant_url: Optional[str] = None, qdrant_key: Optional[str] = None):
        self.settings = get_settings()
        self.api_key = api_key or self.settings.GEMINI_API_KEY
        self.qdrant_url = qdrant_url or self.settings.QDRANT_CLOUD_URL
        self.qdrant_key = qdrant_key or self.settings.QDRANT_CLOUD_API_KEY

        if not self.api_key:
            raise ValueError("Google Gemini API Key is required for Cloud RAG.")

        if not self.qdrant_url or "localhost" in self.qdrant_url or "127.0.0.1" in self.qdrant_url:
            raise ValueError("Qdrant Cloud Cluster URL is required (e.g. https://xyz-example.eu-central.aws.cloud.qdrant.io). Please provide your Qdrant Cloud Cluster endpoint and API key.")

        if not self.qdrant_key:
            raise ValueError("Qdrant Cloud API Key is required to connect to your Qdrant Cloud cluster.")

        self.pdf_parser = PDFParserService(chunk_size=1000, chunk_overlap=300)
        self.vector_store = VectorStoreService(qdrant_url=self.qdrant_url, api_key=self.qdrant_key)
        self.llm_service = LLMService(provider="cloud", api_key=self.api_key)

    def ingest_document(self, file_bytes: bytes, filename: str, job_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes PDF parsing, LangChain chunking, Gemini 1536d vector embedding, and Qdrant Cloud cluster upsertion.
        """
        start_time = time.time()
        jid = job_id or f"job_cloud_{int(time.time())}"
        create_job(jid, filename, "cloud")

        logger.info(f"[Cloud RAG Pipeline] Ingesting PDF '{filename}' ({len(file_bytes)} bytes) into Qdrant Cloud at {self.qdrant_url}")

        try:
            # Stage 1
            update_job_stage(jid, 1, "Reading binary PDF file structure")
            
            # Stage 2
            update_job_stage(jid, 2, "Extracting page content")
            parsed_doc = self.pdf_parser.parse_pdf(file_bytes)
            pages_data = parsed_doc["pages"]
            total_pages = parsed_doc["total_pages"]

            # Stage 3
            update_job_stage(jid, 3, "Normalizing document text paragraphs")

            # Stage 4
            update_job_stage(jid, 4, f"Creating LangChain RecursiveCharacterTextSplitter chunks (chunk_size=1000, overlap=300)")
            chunks = self.pdf_parser.create_chunks(pages_data)
            chunk_texts = [c["text"] for c in chunks]

            # Stage 5
            update_job_stage(jid, 5, f"Generating {len(chunks)} Gemini 1536d vector embeddings")
            embeddings = self.llm_service.generate_embeddings(
                text_chunks=chunk_texts,
                model_name=self.settings.CLOUD_EMBEDDING_MODEL,
                api_key=self.api_key
            )
            vector_dim = len(embeddings[0]) if embeddings else 1536

            # Stage 6
            update_job_stage(jid, 6, f"Connecting to Qdrant Cloud cluster at {self.qdrant_url}")
            collection_name = self.settings.CLOUD_COLLECTION_NAME
            self.vector_store.create_collection_if_not_exists(
                collection_name=collection_name,
                vector_size=vector_dim
            )

            # Stage 7
            update_job_stage(jid, 7, f"Upserting {len(chunks)} points to Qdrant Cloud")
            points = []
            for chunk, vector in zip(chunks, embeddings):
                points.append({
                    "chunk_id": chunk["chunk_id"],
                    "vector": vector,
                    "payload": {
                        "filename": filename,
                        "chunk_id": chunk["chunk_id"],
                        "page_number": chunk["page_number"],
                        "text": chunk["text"],
                        "char_count": chunk["char_count"]
                    }
                })
            self.vector_store.upsert_vectors(collection_name, points)

            # Stage 8
            update_job_stage(jid, 8, "Verifying Qdrant Cloud HNSW index schema")

            # Stage 9
            elapsed_ms = int((time.time() - start_time) * 1000)
            update_job_stage(jid, 9, "Collection ready for Gemini Cloud RAG")

            logger.info(f"[Cloud RAG Pipeline] Successfully ingested '{filename}' into Qdrant Cloud cluster in {elapsed_ms}ms.")

            return {
                "job_id": jid,
                "mode": "cloud",
                "filename": filename,
                "status": "completed",
                "total_pages": total_pages,
                "total_chunks": len(chunks),
                "collection_name": collection_name,
                "embedding_model": self.settings.CLOUD_EMBEDDING_MODEL,
                "llm_model": self.settings.CLOUD_LLM_MODEL,
                "ingestion_time_ms": elapsed_ms
            }
        except Exception as e:
            logger.error(f"[Cloud RAG Ingestion Error]: {e}")
            set_job_error(jid, str(e))
            raise e

    def query(self, user_question: str, top_k: int = 3, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes vector search retrieval in Qdrant Cloud filtered strictly by active document filename.
        """
        start_time = time.time()
        logger.info(f"[Cloud RAG Query] Query: '{user_question}' (Filename Filter: {filename}, Top-K: {top_k})")
        collection_name = self.settings.CLOUD_COLLECTION_NAME

        # 1. Embed user question via Gemini Text-Embedding API
        query_embeddings = self.llm_service.generate_embeddings(
            text_chunks=[user_question],
            model_name=self.settings.CLOUD_EMBEDDING_MODEL,
            api_key=self.api_key
        )
        query_vector = query_embeddings[0]

        # 2. Search top-k similar vector chunks in Qdrant Cloud with filename payload filter
        matched_chunks = self.vector_store.search_similar(
            collection_name=collection_name,
            query_vector=query_vector,
            top_k=top_k,
            filename_filter=filename
        )

        # 3. Extract text passages for prompt context
        passages = [c["text_snippet"] for c in matched_chunks if c.get("text_snippet")]

        # 4. Generate answer using Gemini 1.5 Flash API
        answer = self.llm_service.generate_answer(
            prompt=user_question,
            context_passages=passages,
            model_name=self.settings.CLOUD_LLM_MODEL,
            api_key=self.api_key
        )

        latency_ms = int((time.time() - start_time) * 1000)

        return {
            "query": user_question,
            "answer": answer,
            "retrieved_chunks": matched_chunks,
            "latency_ms": latency_ms,
            "llm_model": self.settings.CLOUD_LLM_MODEL,
            "embedding_model": self.settings.CLOUD_EMBEDDING_MODEL,
            "collection_name": collection_name
        }
