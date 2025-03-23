from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime
from uuid import UUID

class VideoMetrics(BaseModel):
    resolution: Dict[str, int]
    frame_rate: float
    bitrate: float
    color_accuracy: float
    motion_smoothness: float
    artifact_level: float

class AudioMetrics(BaseModel):
    loudness: float
    dynamic_range: float
    frequency_response: Dict[str, float]
    distortion: float
    noise_level: float

class QualityMetrics(BaseModel):
    id: UUID
    job_id: UUID
    video_metrics: VideoMetrics
    audio_metrics: AudioMetrics
    overall_score: float
    created_at: datetime

    class Config:
        orm_mode = True

class ABTest(BaseModel):
    id: UUID
    job_id: UUID
    variant_a: Dict
    variant_b: Dict
    variant_a_metrics: Optional[QualityMetrics]
    variant_b_metrics: Optional[QualityMetrics]
    status: str
    created_at: datetime
    completed_at: Optional[datetime]

    class Config:
        orm_mode = True

class UserFeedback(BaseModel):
    id: UUID
    job_id: UUID
    user_id: UUID
    rating: int = Field(..., ge=1, le=5)
    feedback: Optional[str]
    created_at: datetime

    class Config:
        orm_mode = True

class QualityReport(BaseModel):
    id: UUID
    start_date: datetime
    end_date: datetime
    metrics: Dict
    created_at: datetime

    class Config:
        orm_mode = True

class QualityThreshold(BaseModel):
    id: UUID
    metric_name: str
    threshold_type: str
    threshold_value: float
    severity: str
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        orm_mode = True

class QualityAlert(BaseModel):
    id: UUID
    job_id: UUID
    metric_name: str
    threshold_id: UUID
    actual_value: float
    severity: str
    status: str
    created_at: datetime
    resolved_at: Optional[datetime]

    class Config:
        orm_mode = True

class QualityBenchmark(BaseModel):
    id: UUID
    name: str
    description: Optional[str]
    metrics: Dict
    industry_standard: Optional[Dict]
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        orm_mode = True

class QualityComparison(BaseModel):
    id: UUID
    job_id: UUID
    benchmark_id: UUID
    comparison_metrics: Dict
    created_at: datetime

    class Config:
        orm_mode = True

class QualityAnalysisRequest(BaseModel):
    job_id: UUID
    include_video_metrics: bool = True
    include_audio_metrics: bool = True
    include_overall_score: bool = True

class ABTestRequest(BaseModel):
    job_id: UUID
    variant_a: Dict
    variant_b: Dict

class UserFeedbackRequest(BaseModel):
    job_id: UUID
    rating: int = Field(..., ge=1, le=5)
    feedback: Optional[str]

class QualityReportRequest(BaseModel):
    start_date: datetime
    end_date: datetime
    include_metrics: List[str] = Field(default_factory=list)

class QualityThresholdRequest(BaseModel):
    metric_name: str
    threshold_type: str
    threshold_value: float
    severity: str

class QualityBenchmarkRequest(BaseModel):
    name: str
    description: Optional[str]
    metrics: Dict
    industry_standard: Optional[Dict]

class QualityComparisonRequest(BaseModel):
    job_id: UUID
    benchmark_id: UUID
    metrics_to_compare: List[str] = Field(default_factory=list) 