from typing import Optional, Dict
from pydantic import BaseModel


class ServiceStatus(BaseModel):
    name: str
    status: str  # "running", "error", "pending"
    endpoint: Optional[str] = None
    details: Optional[str] = None


class HealthCheckResponse(BaseModel):
    mode: str  # "local" or "cloud"
    ready: bool
    services: Dict[str, ServiceStatus]
    timestamp: str
