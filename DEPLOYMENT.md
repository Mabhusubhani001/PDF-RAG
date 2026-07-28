# 🚀 PDF RAG Application — Deployment Guide

This guide provides step-by-step instructions to deploy the **PDF RAG Application** across two primary deployment models:

1. **Option A: Free Cloud Hosting** (Vercel + Render/Railway + Qdrant Cloud + Gemini API)
2. **Option B: Self-Hosted Docker Deployment** (Docker Compose + Ollama + Local Qdrant)

---

## ☁️ Option A: Free Cloud Deployment (Recommended)

### Step 1: Deploy Qdrant Cloud Cluster
1. Sign up at [cloud.qdrant.io](https://cloud.qdrant.io).
2. Create a free Tier Cluster.
3. Copy your **Cluster URL** (e.g. `https://xyz-example.eu-central.aws.cloud.qdrant.io:6333`) and **API Key**.

### Step 2: Get Google Gemini API Key
1. Visit [Google AI Studio](https://aistudio.google.com).
2. Generate a free API Key.

### Step 3: Deploy FastAPI Backend (Render or Railway)
1. Push your repository to GitHub.
2. Log in to [Render.com](https://render.com) or [Railway.app](https://railway.app).
3. Create a **New Web Service** pointing to your repository's `/backend` directory.
4. Set Build Command: `pip install -r requirements.txt`
5. Set Start Command: `uvicorn main:app --host 0.0.0.0 --port 8000`
6. Add Environment Variables:
   - `GEMINI_API_KEY` = `<your_gemini_key>`
   - `QDRANT_CLOUD_URL` = `<your_qdrant_cloud_url>`
   - `QDRANT_CLOUD_API_KEY` = `<your_qdrant_cloud_api_key>`
7. Copy your deployed Backend URL (e.g., `https://pdf-rag-backend.onrender.com`).

### Step 4: Deploy Astro Frontend (Vercel or Netlify)
1. Log in to [Vercel.com](https://vercel.com).
2. Import your GitHub repository, selecting the `/pdf-rag` directory.
3. Set Framework Preset: **Astro**.
4. Add Environment Variable:
   - `PUBLIC_BACKEND_URL` = `https://pdf-rag-backend.onrender.com`
5. Click **Deploy**. Your frontend URL will be live!

---

## 🐳 Option B: Self-Hosted Docker Deployment

### Prerequisites
- Install [Docker Desktop](https://www.docker.com/products/docker-desktop).
- Install [Ollama](https://ollama.com) locally and pull models:
  ```bash
  ollama pull llama3.1:latest
  ollama pull qwen3-embedding:0.6b
  ```

### Step 1: Run Qdrant Vector Store
From the root directory or `/backend`, start Qdrant with persistent volume storage:

```bash
# Option 1: Standard Docker run
docker run -d -p 6333:6333 -p 6334:6334 -v ./qdrant_data:/qdrant/storage --name qdrant qdrant/qdrant:latest

# Option 2: Docker Compose DB file
docker compose -f backend/docker-compose.db.yml up -d
```

### Step 2: Launch Full Application via Docker Compose
To run both Qdrant and the FastAPI Backend together:

```bash
docker compose up -d --build
```

Access services at:
- **Astro Frontend**: `http://localhost:4321`
- **FastAPI Backend**: `http://localhost:8000/docs`
- **Qdrant Dashboard**: `http://localhost:6333/dashboard`

---

## 🛠️ Summary of Created Deployment Artifacts

| File | Purpose |
| :--- | :--- |
| [`backend/Dockerfile`](file:///c:/Users/munvar/OneDrive/Desktop/PDF-RAG/backend/Dockerfile) | Production container definition for FastAPI RAG backend |
| [`docker-compose.yml`](file:///c:/Users/munvar/OneDrive/Desktop/PDF-RAG/docker-compose.yml) | Full-stack container compose orchestration |
| [`backend/docker-compose.db.yml`](file:///c:/Users/munvar/OneDrive/Desktop/PDF-RAG/backend/docker-compose.db.yml) | Dedicated Qdrant DB volume compose definition |
| [`DEPLOYMENT.md`](file:///c:/Users/munvar/OneDrive/Desktop/PDF-RAG/DEPLOYMENT.md) | Complete step-by-step deployment manual |
