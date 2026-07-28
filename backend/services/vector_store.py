import httpx
import uuid
from typing import List, Dict, Any, Optional
from utils.logger import logger


class VectorStoreService:
    """
    Strict Qdrant Vector Database service supporting Local Qdrant (Docker) and Qdrant Cloud via REST/gRPC APIs.
    """
    def __init__(self, qdrant_url: str = "http://localhost:6333", api_key: Optional[str] = None):
        self.qdrant_url = qdrant_url.rstrip("/")
        self.api_key = api_key
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["api-key"] = self.api_key

    def check_health(self) -> Dict[str, Any]:
        """
        Ping Qdrant engine health endpoint.
        """
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{self.qdrant_url}/healthz", headers=self.headers)
                if res.status_code == 200:
                    return {"status": "running", "url": self.qdrant_url, "details": "Qdrant active"}
        except Exception as e:
            logger.warning(f"Qdrant connection check failed at {self.qdrant_url}: {e}")
        
        return {"status": "offline", "url": self.qdrant_url, "details": "Qdrant unavailable"}

    def create_collection_if_not_exists(self, collection_name: str, vector_size: int = 1024) -> bool:
        """
        Creates a Qdrant vector collection with Cosine similarity distance metric.
        """
        try:
            with httpx.Client(timeout=5.0) as client:
                check_res = client.get(f"{self.qdrant_url}/collections/{collection_name}", headers=self.headers)
                if check_res.status_code == 200:
                    logger.info(f"Qdrant collection '{collection_name}' already exists.")
                    return True

                payload = {
                    "vectors": {
                        "size": vector_size,
                        "distance": "Cosine"
                    },
                    "hnsw_config": {
                        "m": 16,
                        "ef_construct": 100
                    }
                }
                res = client.put(f"{self.qdrant_url}/collections/{collection_name}", json=payload, headers=self.headers)
                if res.status_code in [200, 201]:
                    logger.info(f"Successfully created Qdrant collection '{collection_name}' ({vector_size}d).")
                    return True
        except Exception as e:
            logger.error(f"Failed to create Qdrant collection '{collection_name}': {e}")
            raise RuntimeError(f"Qdrant Vector DB connection failed at {self.qdrant_url}. Ensure Qdrant credentials are valid.")
        
        return True

    def upsert_vectors(self, collection_name: str, points: List[Dict[str, Any]]) -> bool:
        """
        Upserts dense vectors and payload metadata into Qdrant.
        """
        try:
            with httpx.Client(timeout=15.0) as client:
                formatted_points = []
                for idx, pt in enumerate(points):
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{pt.get('payload', {}).get('filename', 'doc')}_{pt.get('chunk_id', idx)}"))
                    formatted_points.append({
                        "id": point_id,
                        "vector": pt["vector"],
                        "payload": pt["payload"]
                    })

                payload = {"points": formatted_points}
                res = client.put(f"{self.qdrant_url}/collections/{collection_name}/points", json=payload, headers=self.headers)
                if res.status_code in [200, 201]:
                    logger.info(f"Upserted {len(points)} vector points to Qdrant collection '{collection_name}'.")
                    return True
        except Exception as e:
            logger.error(f"Error upserting vectors to Qdrant '{collection_name}': {e}")
            raise RuntimeError(f"Failed to upsert vectors into Qdrant collection '{collection_name}'.")

        return True

    def search_similar(
        self, 
        collection_name: str, 
        query_vector: List[float], 
        top_k: int = 3,
        filename_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes Cosine distance vector search in Qdrant with smart payload filtering and automatic fallback.
        """
        try:
            with httpx.Client(timeout=8.0) as client:
                # 1. First attempt search with filename filter
                if filename_filter:
                    payload: Dict[str, Any] = {
                        "vector": query_vector,
                        "limit": top_k,
                        "with_payload": True,
                        "filter": {
                            "must": [
                                {
                                    "key": "filename",
                                    "match": {
                                        "value": filename_filter
                                    }
                                }
                            ]
                        }
                    }
                    res = client.post(f"{self.qdrant_url}/collections/{collection_name}/points/search", json=payload, headers=self.headers)
                    if res.status_code == 200:
                        results = res.json().get("result", [])
                        if results:
                            return self._parse_qdrant_results(results)

                # 2. Fallback attempt: Standard vector search over the collection if filename filter returns 0 results
                fallback_payload: Dict[str, Any] = {
                    "vector": query_vector,
                    "limit": top_k,
                    "with_payload": True
                }
                res_fallback = client.post(f"{self.qdrant_url}/collections/{collection_name}/points/search", json=fallback_payload, headers=self.headers)
                if res_fallback.status_code == 200:
                    results_fallback = res_fallback.json().get("result", [])
                    return self._parse_qdrant_results(results_fallback)

        except Exception as e:
            logger.error(f"Qdrant search error at {self.qdrant_url}: {e}")

        return []

    def _parse_qdrant_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        matched_chunks = []
        for item in results:
            pld = item.get("payload", {})
            matched_chunks.append({
                "chunk_id": pld.get("chunk_id", "chunk_0"),
                "page_number": pld.get("page_number", 1),
                "filename": pld.get("filename", ""),
                "similarity_score": round(float(item.get("score", 0.0)), 3),
                "text_snippet": pld.get("text", "")
            })
        return matched_chunks
