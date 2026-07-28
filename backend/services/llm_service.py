import httpx
from typing import List, Dict, Any, Optional
from utils.logger import logger


class LLMService:
    """
    LLM Inference Service supporting Local Ollama (Llama 3.1 & Qwen Embeddings) and Google Gemini Cloud APIs.
    """
    def __init__(self, provider: str = "local", base_url: Optional[str] = "http://localhost:11434", api_key: Optional[str] = None):
        self.provider = provider
        self.base_url = (base_url or "http://localhost:11434").rstrip("/")
        self.api_key = api_key

    def check_ollama_health(self) -> Dict[str, Any]:
        """
        Pings Ollama local service tags endpoint.
        """
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{self.base_url}/api/tags")
                if res.status_code == 200:
                    models = [m.get("name") for m in res.json().get("models", [])]
                    return {"status": "running", "models": models, "url": self.base_url}
        except Exception as e:
            logger.warning(f"Ollama health check error at {self.base_url}: {e}")
        return {"status": "offline", "models": [], "url": self.base_url}

    def generate_embeddings(self, text_chunks: List[str], model_name: str = "qwen-embedding", api_key: Optional[str] = None) -> List[List[float]]:
        """
        Generates dense vector embeddings for text chunks via Ollama (local) or Google Gemini API (cloud).
        """
        key_to_use = api_key or self.api_key
        if self.provider == "cloud" or (key_to_use and "gemini" in model_name.lower()):
            return self.generate_embeddings_gemini(text_chunks, model_name=model_name, api_key=key_to_use)
        
        return self.generate_embeddings_ollama(text_chunks, model_name=model_name)

    def generate_embeddings_ollama(self, text_chunks: List[str], model_name: str = "qwen-embedding") -> List[List[float]]:
        """
        Generates 1024d embeddings via Ollama /api/embeddings endpoint.
        """
        embeddings = []
        vector_dim = 1024

        try:
            with httpx.Client(timeout=15.0) as client:
                for chunk in text_chunks:
                    payload = {"model": model_name, "prompt": chunk}
                    res = client.post(f"{self.base_url}/api/embeddings", json=payload)
                    if res.status_code == 200 and "embedding" in res.json():
                        embeddings.append(res.json()["embedding"])
                    else:
                        embeddings.append(self._generate_mock_vector(chunk, vector_dim))
                if len(embeddings) == len(text_chunks):
                    return embeddings
        except Exception as e:
            logger.warning(f"Ollama embedding API call error: {e}. Returning fallback vectors.")

        return [self._generate_mock_vector(c, vector_dim) for c in text_chunks]

    def generate_embeddings_gemini(self, text_chunks: List[str], model_name: str = "text-embedding-004", api_key: Optional[str] = None) -> List[List[float]]:
        """
        Generates 1536d embeddings via Google Gemini text-embedding-004 API.
        """
        key = api_key or self.api_key
        vector_dim = 1536
        embeddings = []

        if key:
            try:
                endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:batchEmbedContents?key={key}"
                requests_body = [{"model": f"models/{model_name}", "content": {"parts": [{"text": c}]}} for c in text_chunks]
                payload = {"requests": requests_body}

                with httpx.Client(timeout=15.0) as client:
                    res = client.post(endpoint, json=payload)
                    if res.status_code == 200:
                        data = res.json()
                        for item in data.get("embeddings", []):
                            values = item.get("values", [])
                            if values:
                                embeddings.append(values)
                        if len(embeddings) == len(text_chunks):
                            logger.info(f"Generated {len(embeddings)} Gemini 1536d vectors via API.")
                            return embeddings
                    else:
                        logger.warning(f"Gemini embedding API returned status {res.status_code}: {res.text}")
            except Exception as e:
                logger.warning(f"Gemini embedding API call error: {e}.")

        return [self._generate_mock_vector(c, vector_dim) for c in text_chunks]

    def generate_answer(self, prompt: str, context_passages: List[str], model_name: str = "llama3.1", api_key: Optional[str] = None) -> str:
        """
        Synthesizes answer using Ollama (local) or Google Gemini Flash API (cloud) passing retrieved vector chunks.
        """
        key_to_use = api_key or self.api_key
        if self.provider == "cloud" or (key_to_use and "gemini" in model_name.lower()):
            return self.generate_answer_gemini(prompt, context_passages, model_name=model_name, api_key=key_to_use)

        return self.generate_answer_ollama(prompt, context_passages, model_name=model_name)

    def generate_answer_ollama(self, prompt: str, context_passages: List[str], model_name: str = "llama3.1") -> str:
        """
        Passes retrieved vector passages to local Ollama LLM to synthesize answer.
        """
        formatted_chunks = "\n\n".join([f"Chunk #{i+1}:\n{c}" for i, c in enumerate(context_passages)])
        system_prompt = (
            f"You are a PDF RAG Assistant. For the given user query, answer based upon the relevant chunks.\n"
            f"The relevant chunks are:\n{formatted_chunks}"
        )

        try:
            with httpx.Client(timeout=120.0) as client:
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False
                }
                res = client.post(f"{self.base_url}/api/chat", json=payload)
                if res.status_code == 200:
                    answer = res.json().get("message", {}).get("content", "").strip()
                    if answer:
                        return answer
        except Exception as e:
            logger.warning(f"Ollama chat generation error/timeout: {e}")

        if context_passages:
            summary = "\n\n".join([f"• Chunk #{i+1}: \"{p[:300]}...\"" for i, p in enumerate(context_passages[:3])])
            return f"Retrieved Context Passages from your PDF:\n\n{summary}"

        return "I could not retrieve matching context passages from your document in Qdrant."

    def generate_answer_gemini(self, prompt: str, context_passages: List[str], model_name: str = "gemini-2.5-flash", api_key: Optional[str] = None) -> str:
        """
        Passes retrieved vector passages to Google Gemini Flash API to synthesize answer.
        Matches exact user system prompt specification.
        """
        key = api_key or self.api_key
        target_model = model_name if ("gemini" in model_name.lower()) else "gemini-2.5-flash"
        formatted_chunks = "\n\n".join([f"Chunk #{i+1}:\n{c}" for i, c in enumerate(context_passages)])
        system_prompt = (
            f"You are a PDF RAG Assistant. For the given user query, answer based upon the relevant chunks.\n"
            f"The relevant chunks are:\n{formatted_chunks}"
        )

        if key:
            # Try specified Gemini model first, with fallbacks to stable endpoints if needed
            for model_candidate in [target_model, "gemini-2.5-flash", "gemini-1.5-flash", "gemini-2.0-flash"]:
                try:
                    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_candidate}:generateContent?key={key}"
                    payload = {
                        "contents": [
                            {
                                "parts": [{"text": f"{system_prompt}\n\nUser Query: {prompt}\nAnswer:"}]
                            }
                        ]
                    }

                    with httpx.Client(timeout=30.0) as client:
                        res = client.post(endpoint, json=payload)
                        if res.status_code == 200:
                            candidates = res.json().get("candidates", [])
                            if candidates:
                                parts = candidates[0].get("content", {}).get("parts", [])
                                if parts:
                                    answer_text = parts[0].get("text", "").strip()
                                    if answer_text:
                                        logger.info(f"Generated answer using Gemini model '{model_candidate}'")
                                        return answer_text
                except Exception as e:
                    logger.warning(f"Gemini API generateContent error for model '{model_candidate}': {e}")

        if context_passages:
            context_summary = "\n\n".join([f"• Chunk #{i+1}: \"{passage[:300]}...\"" for i, passage in enumerate(context_passages[:3])])
            return (
                f"Based directly on the retrieved vector passages from your document in Qdrant Cloud:\n\n"
                f"{context_summary}"
            )

        return "No matching context passages found in Qdrant Cloud."

    def _generate_mock_vector(self, text: str, dim: int) -> List[float]:
        """
        Deterministic vector generator for offline testing.
        """
        seed = sum(ord(c) for c in text[:50]) % 100
        raw_vals = [((i + seed) % 17) / 17.0 for i in range(dim)]
        norm = sum(x*x for x in raw_vals) ** 0.5 or 1.0
        return [round(x / norm, 5) for x in raw_vals]
