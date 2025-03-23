from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from .database import JobType, JobStatus

class JobBase(BaseModel):
    """Base job model."""
    job_type: JobType
    input_data: Dict[str, Any]

class JobCreate(JobBase):
    """Job creation model."""
    user_id: Optional[str] = None  # Will be set by the API endpoint

class JobUpdate(BaseModel):
    """Job update model."""
    status: Optional[JobStatus] = None
    progress: Optional[float] = Field(None, ge=0.0, le=100.0)
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class JobResponse(JobBase):
    """Job response model."""
    id: str
    job_id: str
    user_id: str
    status: JobStatus
    progress: float = Field(ge=0.0, le=100.0)
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        orm_mode = True

class JobListResponse(BaseModel):
    """Job list response model."""
    jobs: list[JobResponse]
    total: int
    skip: int
    limit: int 