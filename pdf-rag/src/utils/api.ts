/**
 * REST Client for communicating with the FastAPI backend engine.
 * Endpoint URL is configurable via import.meta.env.PUBLIC_BACKEND_URL or defaults to http://localhost:8000.
 */

const BACKEND_URL = (import.meta.env.PUBLIC_BACKEND_URL || 'http://localhost:8000').replace(/\/$/, '');
const API_BASE = `${BACKEND_URL}/api/v1`;

export interface ServiceStatus {
  name: string;
  status: 'running' | 'error' | 'pending';
  endpoint?: string;
  details?: string;
}

export interface HealthCheckResponse {
  mode: string;
  ready: boolean;
  services: Record<string, ServiceStatus>;
  timestamp: string;
}

export interface DocumentMetadata {
  filename: string;
  file_size_bytes: number;
  content_type: string;
  estimated_pages: number;
  estimated_chunks: number;
}

export interface UploadResponse {
  job_id: string;
  filename: string;
  status: string;
  metadata: DocumentMetadata;
  message: string;
}

export interface PipelineStageInfo {
  stage_number: number;
  name: string;
  description: string;
  status: 'queued' | 'in_progress' | 'completed' | 'error';
  execution_time_ms?: number;
}

export interface PipelineProgressResponse {
  job_id: string;
  mode: string;
  current_stage: number;
  total_stages: number;
  percent_complete: number;
  is_completed: boolean;
  stages: PipelineStageInfo[];
}

export interface RetrievedChunk {
  chunk_id: string;
  page_number: number;
  similarity_score: number;
  text_snippet: string;
  filename?: string;
  metadata?: Record<string, any>;
}

export interface ChatQueryResponse {
  query: string;
  answer: string;
  retrieved_chunks: RetrievedChunk[];
  latency_ms: number;
  llm_model: string;
  embedding_model: string;
  collection_name: string;
}

function getCloudHeaders(): Record<string, string> {
  const headers: Record<string, string> = {};
  if (typeof window !== 'undefined') {
    const geminiKey = sessionStorage.getItem('gemini_api_key');
    const qdrantUrl = sessionStorage.getItem('qdrant_cloud_url');
    const qdrantKey = sessionStorage.getItem('qdrant_cloud_key');

    if (geminiKey) headers['x-gemini-api-key'] = geminiKey;
    if (qdrantUrl) headers['x-qdrant-url'] = qdrantUrl;
    if (qdrantKey) headers['x-qdrant-api-key'] = qdrantKey;
  }
  return headers;
}

/**
 * Fetch system readiness health status.
 */
export async function fetchSystemHealth(): Promise<HealthCheckResponse | null> {
  try {
    const res = await fetch(`${API_BASE}/health`, { method: 'GET' });
    if (res.ok) {
      return await res.json();
    }
  } catch (err) {
    console.warn('[API Client] Health check failed:', err);
  }
  return null;
}

/**
 * Upload PDF file to backend for parsing and vector embedding.
 */
export async function uploadPdfDocument(file: File, mode: 'local' | 'cloud'): Promise<UploadResponse | null> {
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('mode', mode);

    const headers = getCloudHeaders();

    const res = await fetch(`${API_BASE}/upload`, {
      method: 'POST',
      headers: headers,
      body: formData,
    });

    if (res.ok) {
      return await res.json();
    }
  } catch (err) {
    console.warn('[API Client] Upload request failed:', err);
  }
  return null;
}

/**
 * Fetch live 9-stage pipeline progress for a job.
 */
export async function fetchPipelineProgress(jobId: string, mode: 'local' | 'cloud'): Promise<PipelineProgressResponse | null> {
  try {
    const res = await fetch(`${API_BASE}/pipeline/${jobId}/status?mode=${mode}`, { method: 'GET' });
    if (res.ok) {
      return await res.json();
    }
  } catch (err) {
    console.warn('[API Client] Pipeline status check failed:', err);
  }
  return null;
}

/**
 * Send user RAG chat query to backend with optional active filename filter.
 */
export async function queryRagChat(query: string, mode: 'local' | 'cloud', topK: number = 3, filename?: string): Promise<ChatQueryResponse | null> {
  try {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...getCloudHeaders(),
    };

    const res = await fetch(`${API_BASE}/chat/query`, {
      method: 'POST',
      headers: headers,
      body: JSON.stringify({ query, mode, top_k: topK, filename }),
    });

    if (res.ok) {
      return await res.json();
    }
  } catch (err) {
    console.warn('[API Client] Chat query failed:', err);
  }
  return null;
}
