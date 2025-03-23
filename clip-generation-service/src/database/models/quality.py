from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, JSON, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from .base import Base

class QualityMetricsDB(Base):
    __tablename__ = "quality_metrics"

    id = Column(String(36), primary_key=True)
    job_id = Column(String(36), ForeignKey("jobs.id"), nullable=False)
    video_metrics = Column(JSON, nullable=False)
    audio_metrics = Column(JSON, nullable=False)
    overall_score = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    job = relationship("JobDB", back_populates="quality_metrics")

class ABTestDB(Base):
    __tablename__ = "ab_tests"

    id = Column(String(36), primary_key=True)
    job_id = Column(String(36), ForeignKey("jobs.id"), nullable=False)
    variant_a = Column(JSON, nullable=False)
    variant_b = Column(JSON, nullable=False)
    variant_a_metrics = Column(JSON)
    variant_b_metrics = Column(JSON)
    status = Column(String(20), nullable=False)  # active, completed, cancelled
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))

    job = relationship("JobDB", back_populates="ab_tests")

class UserFeedbackDB(Base):
    __tablename__ = "user_feedback"

    id = Column(String(36), primary_key=True)
    job_id = Column(String(36), ForeignKey("jobs.id"), nullable=False)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    rating = Column(Integer, nullable=False)  # 1-5 scale
    feedback = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    job = relationship("JobDB", back_populates="user_feedback")
    user = relationship("UserDB", back_populates="feedback")

class QualityReportDB(Base):
    __tablename__ = "quality_reports"

    id = Column(String(36), primary_key=True)
    start_date = Column(DateTime(timezone=True), nullable=False)
    end_date = Column(DateTime(timezone=True), nullable=False)
    metrics = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class QualityThresholdDB(Base):
    __tablename__ = "quality_thresholds"

    id = Column(String(36), primary_key=True)
    metric_name = Column(String(100), nullable=False)
    threshold_type = Column(String(20), nullable=False)  # min, max, range
    threshold_value = Column(Float, nullable=False)
    severity = Column(String(20), nullable=False)  # warning, error, critical
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class QualityAlertDB(Base):
    __tablename__ = "quality_alerts"

    id = Column(String(36), primary_key=True)
    job_id = Column(String(36), ForeignKey("jobs.id"), nullable=False)
    metric_name = Column(String(100), nullable=False)
    threshold_id = Column(String(36), ForeignKey("quality_thresholds.id"), nullable=False)
    actual_value = Column(Float, nullable=False)
    severity = Column(String(20), nullable=False)
    status = Column(String(20), nullable=False)  # active, resolved, dismissed
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    resolved_at = Column(DateTime(timezone=True))

    job = relationship("JobDB", back_populates="quality_alerts")
    threshold = relationship("QualityThresholdDB")

class QualityBenchmarkDB(Base):
    __tablename__ = "quality_benchmarks"

    id = Column(String(36), primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    metrics = Column(JSON, nullable=False)
    industry_standard = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class QualityComparisonDB(Base):
    __tablename__ = "quality_comparisons"

    id = Column(String(36), primary_key=True)
    job_id = Column(String(36), ForeignKey("jobs.id"), nullable=False)
    benchmark_id = Column(String(36), ForeignKey("quality_benchmarks.id"), nullable=False)
    comparison_metrics = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    job = relationship("JobDB", back_populates="quality_comparisons")
    benchmark = relationship("QualityBenchmarkDB") 