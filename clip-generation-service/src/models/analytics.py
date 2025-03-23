from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel

class UserAnalytics(BaseModel):
    user_id: str
    total_jobs: int
    completed_jobs: int
    success_rate: float
    total_tokens_used: int
    average_tokens_per_job: float
    total_processing_time: float
    most_used_features: List[str]
    recent_sessions: List[Dict[str, Optional[datetime | float]]]

class FeatureUsage(BaseModel):
    feature_name: str
    total_usage: int
    usage_24h: int
    usage_7d: int
    usage_30d: int
    growth_rate: float

class UserEngagement(BaseModel):
    user_id: str
    total_sessions: int
    total_duration: float
    average_session_duration: float
    feature_interactions: Dict[str, int]

class TokenConsumption(BaseModel):
    user_id: str
    daily_usage: Dict[str, int]
    feature_usage: Dict[str, int]

class ConversionMetrics(BaseModel):
    total_users: int
    active_users: int
    subscription_rate: float
    feature_adoption: Dict[str, int]

class SystemMetrics(BaseModel):
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    uptime: float
    queue_length: int
    processing_backlog: int
    error_rate: float
    average_processing_time: float

class APIMetrics(BaseModel):
    total_requests: int
    error_requests: int
    error_rate: float
    average_latency: float
    endpoint_metrics: Dict[str, Dict[str, float]]

class DatabaseMetrics(BaseModel):
    total_connections: int
    active_connections: int
    connection_pool_size: int
    query_latency: Dict[str, float]
    error_rate: float

class CacheMetrics(BaseModel):
    total_requests: int
    hit_rate: float
    miss_rate: float
    memory_usage: float
    eviction_rate: float

class AnalyticsReport(BaseModel):
    timestamp: datetime
    system: SystemMetrics
    api: APIMetrics
    database: DatabaseMetrics
    cache: CacheMetrics
    user_metrics: Dict[str, UserAnalytics]
    feature_usage: Dict[str, FeatureUsage]
    conversion: ConversionMetrics 