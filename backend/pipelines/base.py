from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseRAGPipeline(ABC):
    """
    Abstract Base Class defining the contract for RAG Pipelines (Local vs. Cloud).
    """

    @abstractmethod
    def ingest_document(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Executes document loading, splitting, embedding, and vector DB insertion.
        """
        pass

    @abstractmethod
    def query(self, user_question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Executes semantic search retrieval and LLM answer synthesis.
        """
        pass
