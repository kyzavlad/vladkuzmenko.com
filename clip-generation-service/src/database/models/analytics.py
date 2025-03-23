from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey, JSON, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from .base import Base

class APIRequestDB(Base):
    __tablename__ = "api_requests"

    id = Column(String(36), primary_key=True)
    endpoint = Column(String(100), nullable=False)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer, nullable=False)
    latency = Column(Float, nullable=False)
    user_id = Column(String(36), ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("UserDB", back_populates="api_requests")

class UserSessionDB(Base):
    __tablename__ = "user_sessions"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    end_time = Column(DateTime(timezone=True))
    duration = Column(Float)
    user_agent = Column(String(200))
    ip_address = Column(String(45))
    location = Column(JSON)

    user = relationship("UserDB", back_populates="sessions")

class FeatureUsageDB(Base):
    __tablename__ = "feature_usage"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    feature_name = Column(String(100), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    metadata = Column(JSON)

    user = relationship("UserDB", back_populates="feature_usage")

class SystemMetricsDB(Base):
    __tablename__ = "system_metrics"

    id = Column(String(36), primary_key=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    cpu_usage = Column(Float, nullable=False)
    memory_usage = Column(Float, nullable=False)
    disk_usage = Column(Float, nullable=False)
    queue_length = Column(Integer, nullable=False)
    processing_backlog = Column(Integer, nullable=False)
    error_rate = Column(Float, nullable=False)
    average_processing_time = Column(Float, nullable=False)

class DatabaseMetricsDB(Base):
    __tablename__ = "database_metrics"

    id = Column(String(36), primary_key=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    total_connections = Column(Integer, nullable=False)
    active_connections = Column(Integer, nullable=False)
    connection_pool_size = Column(Integer, nullable=False)
    query_latency = Column(JSON, nullable=False)
    error_rate = Column(Float, nullable=False)

class CacheMetricsDB(Base):
    __tablename__ = "cache_metrics"

    id = Column(String(36), primary_key=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    total_requests = Column(Integer, nullable=False)
    hit_rate = Column(Float, nullable=False)
    miss_rate = Column(Float, nullable=False)
    memory_usage = Column(Float, nullable=False)
    eviction_rate = Column(Float, nullable=False)

class AnalyticsReportDB(Base):
    __tablename__ = "analytics_reports"

    id = Column(String(36), primary_key=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    report_type = Column(String(50), nullable=False)  # daily, weekly, monthly
    start_date = Column(DateTime(timezone=True), nullable=False)
    end_date = Column(DateTime(timezone=True), nullable=False)
    metrics = Column(JSON, nullable=False)
    summary = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now()) 