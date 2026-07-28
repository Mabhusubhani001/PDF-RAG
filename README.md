<div align="center">

# 📄 PDF RAG — Advanced Visual PDF Retrieval-Augmented Generation Platform

**A full-stack, dual-engine RAG platform to chat with PDF documents locally via Ollama + Qdrant or in the cloud via Google Gemini + Qdrant Cloud.**

Developed by **Shaik Mabhu Subhani**

---

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Astro](https://img.shields.io/badge/Astro-BC52EE?style=for-the-badge&logo=astro&logoColor=white)](https://astro.build/)
[![Qdrant](https://img.shields.io/badge/Qdrant-D62828?style=for-the-badge&logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://python.langchain.com/)
[![Ollama](https://img.shields.io/badge/Ollama-black?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)](https://ai.google.dev/)

</div>

---

## 🌟 Overview

**PDF RAG** is a production-grade Retrieval-Augmented Generation (RAG) platform designed to ingest PDF documents, parse visual layouts, perform sentence-aware semantic chunking, generate high-dimensional dense vector embeddings, and synthesize accurate, cited answers using AI language models.

The system supports two execution pipelines:
1. 🏠 **Local PDF RAG (Ollama + Local Qdrant)**: 100% offline privacy using Ollama (`llama3.1:latest` + `qwen3-embedding:0.6b`) & Local Qdrant Docker container on port `6333`.
2. ☁️ **Cloud PDF RAG (Google Gemini + Qdrant Cloud)**: High-speed cloud inference using Google Gemini (`text-embedding-004` + `gemini-2.5-flash` / `gemini-1.5-flash`) & managed Qdrant Cloud clusters.

---

## 🚀 Key Features

- 🎯 **LangChain Semantic Chunking**: Powered by `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)` for clean, sentence-aware chunk boundaries with zero broken words.
- 📊 **Visual 9-Stage Indexing Pipeline**: Real-time progress bar and animated status badges tracking every ingestion step from PDF parsing to HNSW graph index construction.
- 🔍 **Retrieval Chunk Inspector**: Inspect exact retrieved vector passages, page numbers, and cosine similarity scores.
- 🔒 **Document Payload Isolation**: Qdrant payload filters isolate vector retrieval strictly to the currently active document, preventing cross-document context contamination.
- 💰 **Bring-Your-Own-Key (BYOK)**: Zero hosting costs for developers—users enter their own free Gemini API key and Qdrant Cloud cluster URL.

---

## 🏗️ System Architecture

```
                               ┌───────────────────────────┐
                               │     Astro 4 Frontend      │
                               │  (HTML5 / Tailwind CSS)   │
                               └─────────────┬─────────────┘
                                             │ HTTP REST / JSON
                               ┌─────────────▼─────────────┐
                               │   FastAPI Python Engine   │
                               └──────┬─────────────┬──────┘
                                      │             │
                    ┌─────────────────┘             └─────────────────┐
                    ▼                                                 ▼
     【 LOCAL RAG PIPELINE 】                              【 CLOUD RAG PIPELINE 】
 ┌───────────────────────────────┐                 ┌───────────────────────────────┐
 │ PyMuPDF + LangChain Splitter  │                 │ PyMuPDF + LangChain Splitter  │
 │ Ollama (qwen3-embedding 1024d)│                 │ Gemini text-embedding-004     │
 │ Qdrant Docker (localhost:6333)│                 │ Qdrant Cloud Cluster (TLS)    │
 │ Ollama LLM (llama3.1:latest)  │                 │ Gemini 2.5 Flash / 1.5 Flash  │
 └───────────────────────────────┘                 └───────────────────────────────┘
```

---

## 🛠️ Tech Stack

- **Frontend**: Astro 4, Tailwind CSS 4, TypeScript, HTML5
- **Backend**: FastAPI, PyMuPDF, LangChain (`langchain-text-splitters`), `pypdf`, `httpx`
- **Vector Database**: Qdrant Vector Database (Cosine Distance Metric, HNSW Graph Indexing)
- **AI Models**:
  - Local Embeddings: `qwen3-embedding:0.6b` (1024d)
  - Local LLM: `llama3.1:latest`
  - Cloud Embeddings: `text-embedding-004` (1536d)
  - Cloud LLM: `gemini-2.5-flash` / `gemini-1.5-flash`

---

## 💻 Quick Start (Running Locally)

### 1. Prerequisites
- Python 3.12+
- Node.js 18+
- Docker Desktop
- Ollama running locally:
  ```bash
  ollama pull llama3.1:latest
  ollama pull qwen3-embedding:0.6b
  ```

### 2. Start Qdrant Vector Store
```bash
docker run -d -p 6333:6333 -p 6334:6334 -v ./qdrant_data:/qdrant/storage --name qdrant qdrant/qdrant:latest
```

### 3. Start Backend Server
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 4. Start Frontend Website
```bash
cd pdf-rag
npm install
npm run dev
```

Open **`http://localhost:4321`** in your browser!

---

## 📡 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | System health & readiness status |
| `POST` | `/api/v1/upload` | Parse PDF, create LangChain chunks, embed & index into Qdrant |
| `GET` | `/api/v1/pipeline/{job_id}/status` | Poll live 9-stage pipeline progress |
| `POST` | `/api/v1/chat/query` | Perform vector search retrieval & LLM answer synthesis |

---

## 👤 Author

**Shaik Mabhu Subhani**  
- GitHub: [@Mabhusubhani001](https://github.com/Mabhusubhani001)

---

## 📝 License

Distributed under the MIT License.
