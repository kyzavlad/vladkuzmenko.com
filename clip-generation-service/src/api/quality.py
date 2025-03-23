from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
import uuid

from ..database.session import get_db
from ..services.quality_control import QualityControlService
from ..models.quality import (
    QualityMetrics,
    ABTest,
    UserFeedback,
    QualityReport,
    QualityThreshold,
    QualityAlert,
    QualityBenchmark,
    QualityComparison
)
from ..auth import get_current_user
from ..models.user import User

router = APIRouter(
    prefix="/quality",
    tags=["quality"]
)

def get_quality_service(db: Session = Depends(get_db)) -> QualityControlService:
    return QualityControlService(db)

@router.post("/analyze/{job_id}", response_model=QualityMetrics)
async def analyze_job_quality(
    job_id: str,
    quality_service: QualityControlService = Depends(get_quality_service)
):
    """Analyze the quality of a processed job."""
    return await quality_service.analyze_output_quality(job_id)

@router.post("/ab-test/{job_id}", response_model=ABTest)
async def create_ab_test(
    job_id: str,
    variant_a: dict,
    variant_b: dict,
    quality_service: QualityControlService = Depends(get_quality_service)
):
    """Create an A/B test for a job."""
    return await quality_service.setup_ab_test(job_id, variant_a, variant_b)

@router.post("/ab-test/{test_id}/record", response_model=ABTest)
async def record_ab_test_result(
    test_id: str,
    variant_a_metrics: dict,
    variant_b_metrics: dict,
    quality_service: QualityControlService = Depends(get_quality_service)
):
    """Record results for an A/B test."""
    return await quality_service.record_ab_test_result(
        test_id, variant_a_metrics, variant_b_metrics
    )

@router.get("/ab-test/{test_id}/results", response_model=dict)
async def get_ab_test_results(
    test_id: str,
    quality_service: QualityControlService = Depends(get_quality_service)
):
    """Get results of an A/B test."""
    return await quality_service.analyze_ab_test_results(test_id)

@router.post("/feedback/{job_id}", response_model=UserFeedback)
async def submit_feedback(
    job_id: str,
    rating: int,
    feedback: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    quality_service: QualityControlService = Depends(get_quality_service)
):
    """Submit user feedback for a job."""
    return await quality_service.collect_user_feedback(
        job_id, current_user.id, rating, feedback
    )

@router.get("/report", response_model=QualityReport)
async def generate_quality_report(
    start_date: datetime = Query(...),
    end_date: datetime = Query(...),
    quality_service: QualityControlService = Depends(get_quality_service)
):
    """Generate a quality report for a time period."""
    return await quality_service.generate_quality_report(start_date, end_date)

@router.get("/alerts", response_model=List[QualityAlert])
async def get_quality_alerts(
    status: Optional[str] = None,
    severity: Optional[str] = None,
    quality_service: QualityControlService = Depends(get_quality_service)
):
    """Get quality alerts with optional filtering."""
    return await quality_service.get_quality_alerts(status, severity)

@router.post("/thresholds", response_model=QualityThreshold)
async def create_quality_threshold(
    metric_name: str,
    threshold_type: str,
    threshold_value: float,
    severity: str,
    quality_service: QualityControlService = Depends(get_quality_service)
):
    """Create a new quality threshold."""
    return await quality_service.create_quality_threshold(
        metric_name, threshold_type, threshold_value, severity
    )

@router.get("/thresholds", response_model=List[QualityThreshold])
async def get_quality_thresholds(
    quality_service: QualityControlService = Depends(get_quality_service)
):
    """Get all quality thresholds."""
    return await quality_service.get_quality_thresholds()

@router.post("/benchmarks", response_model=QualityBenchmark)
async def create_quality_benchmark(
    name: str,
    description: Optional[str] = None,
    metrics: dict = None,
    industry_standard: Optional[dict] = None,
    quality_service: QualityControlService = Depends(get_quality_service)
):
    """Create a new quality benchmark."""
    return await quality_service.create_quality_benchmark(
        name, description, metrics, industry_standard
    )

@router.get("/benchmarks", response_model=List[QualityBenchmark])
async def get_quality_benchmarks(
    quality_service: QualityControlService = Depends(get_quality_service)
):
    """Get all quality benchmarks."""
    return await quality_service.get_quality_benchmarks()

@router.post("/compare/{job_id}", response_model=QualityComparison)
async def compare_with_benchmark(
    job_id: str,
    benchmark_id: str,
    quality_service: QualityControlService = Depends(get_quality_service)
):
    """Compare a job's quality against a benchmark."""
    return await quality_service.compare_with_benchmark(job_id, benchmark_id)

@router.get("/metrics/{job_id}", response_model=QualityMetrics)
async def get_job_quality_metrics(
    job_id: str,
    quality_service: QualityControlService = Depends(get_quality_service)
):
    """Get quality metrics for a specific job."""
    return await quality_service.get_job_quality_metrics(job_id)

@router.get("/degradation", response_model=List[dict])
async def check_quality_degradation(
    metric_name: Optional[str] = None,
    time_window: Optional[int] = 24,  # hours
    quality_service: QualityControlService = Depends(get_quality_service)
):
    """Check for quality degradation in metrics."""
    return await quality_service.detect_quality_degradation(metric_name, time_window) 