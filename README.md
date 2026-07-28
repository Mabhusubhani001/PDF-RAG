<div align="center">

# 📄 PDF RAG — Visual PDF Retrieval-Augmented Generation Platform

**A full-stack, dual-mode RAG application to chat with PDF documents locally via Ollama + Qdrant or in the cloud via Google Gemini + Qdrant Cloud.**

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Astro](https://img.shields.io/badge/Astro-BC52EE?style=for-the-badge&logo=astro&logoColor=white)](https://astro.build/)
[![Qdrant](https://img.shields.io/badge/Qdrant-D62828?style=for-the-badge&logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://python.langchain.com/)
[![Ollama](https://img.shields.io/badge/Ollama-black?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)](https://ai.google.dev/)

</div>

---

## ✨ Features

- ⚡ **Dual RAG Architecture**:
  - **Local RAG Mode**: 100% offline privacy using Ollama (`llama3.1` + `qwen3-embedding`) & Local Qdrant Docker container.
  - **Cloud RAG Mode**: Ultra-fast cloud inference using Google Gemini (`text-embedding-004` + `gemini-2.5-flash`) & Qdrant Cloud.
- 🎯 **LangChain Chunking Engine**: Powered by `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)` for clean, sentence-aware chunk boundaries with zero broken words.
- 📊 **Visual 9-Stage Indexing Pipeline**: Real-time progress bar and animated status pills tracking every ingestion step from PDF parsing to HNSW graph index construction.
- 🔍 **Retrieval Chunk Inspector**: Inspect exact retrieved vector passages, page numbers, and cosine similarity scores.
- 🔒 **Document Payload Filtering**: Qdrant payload filters isolate vector retrieval to the currently active document, preventing cross-document context contamination.
- 💰 **Bring-Your-Own-Key (BYOK)**: Zero hosting costs for developers—users provide their own free Gemini API key and Qdrant Cloud cluster URL.

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

## 🚀 Quick Start (Running Locally)

### Prerequisites
- **Python**: 3.12+
- **Node.js**: 18+
- **Docker Desktop**: For running local Qdrant container
- **Ollama**: Running locally with models:
  ```bash
  ollama pull llama3.1:latest
  ollama pull qwen3-embedding:0.6b
  ```

---

### 1. Start Qdrant Vector Store Container

```bash
docker run -d -p 6333:6333 -p 6334:6334 -v ./qdrant_data:/qdrant/storage --name qdrant qdrant/qdrant:latest
```

---

### 2. Start FastAPI Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

- **Backend API**: `http://localhost:8000`
- **Swagger Documentation**: `http://localhost:8000/docs`

---

### 3. Start Astro Frontend

```bash
cd pdf-rag
npm install
npm run dev
```

- **Frontend Application**: `http://localhost:4321`

---

## 📡 REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | System readiness & service status ping |
| `POST` | `/api/v1/upload` | Parse PDF, create LangChain chunks, generate embeddings & index into Qdrant |
| `GET` | `/api/v1/pipeline/{job_id}/status` | Poll live 9-stage visual pipeline progress |
| `POST` | `/api/v1/chat/query` | Execute RAG vector search retrieval & LLM answer synthesis |

---

## 🌐 Deployment

For complete instructions on deploying to **Vercel**, **Render**, or **Docker Compose**, check out the **[DEPLOYMENT.md](file:///c:/Users/munvar/OneDrive/Desktop/PDF-RAG/DEPLOYMENT.md)** guide.

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.
