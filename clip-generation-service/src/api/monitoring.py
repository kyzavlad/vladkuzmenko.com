from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from ..database.session import get_db
from ..services.monitoring_service import MonitoringService
from ..services.analytics_service import AnalyticsService
from ..models.analytics import (
    UserAnalytics, FeatureUsage, UserEngagement,
    TokenConsumption, ConversionMetrics, AnalyticsReport
)
from ..auth import get_current_user
from ..models.user import User

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

def get_monitoring_service(db: Session = Depends(get_db)) -> MonitoringService:
    return MonitoringService(db)

def get_analytics_service(db: Session = Depends(get_db)) -> AnalyticsService:
    return AnalyticsService(db)

# System Monitoring Endpoints
@router.get("/system/metrics")
async def get_system_metrics(
    monitoring_service: MonitoringService = Depends(get_monitoring_service)
):
    """Get current system metrics."""
    return await monitoring_service.get_all_metrics()

@router.get("/system/metrics/history")
async def get_system_metrics_history(
    start_time: datetime,
    end_time: datetime,
    monitoring_service: MonitoringService = Depends(get_monitoring_service)
):
    """Get historical system metrics."""
    # TODO: Implement historical metrics retrieval
    return {"message": "Historical metrics endpoint"}

# User Analytics Endpoints
@router.get("/analytics/user/{user_id}", response_model=UserAnalytics)
async def get_user_analytics(
    user_id: str,
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """Get analytics for a specific user."""
    if not current_user.is_admin and str(current_user.id) != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this user's analytics"
        )
    return await analytics_service.get_user_analytics(user_id)

@router.get("/analytics/feature/{feature_name}", response_model=FeatureUsage)
async def get_feature_usage(
    feature_name: str,
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """Get usage statistics for a specific feature."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return await analytics_service.get_feature_usage(feature_name)

@router.get("/analytics/engagement/{user_id}", response_model=UserEngagement)
async def get_user_engagement(
    user_id: str,
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """Get user engagement metrics."""
    if not current_user.is_admin and str(current_user.id) != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this user's engagement metrics"
        )
    return await analytics_service.get_user_engagement(user_id)

@router.get("/analytics/tokens/{user_id}", response_model=TokenConsumption)
async def get_token_consumption(
    user_id: str,
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """Get token consumption patterns for a user."""
    if not current_user.is_admin and str(current_user.id) != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this user's token consumption"
        )
    return await analytics_service.get_token_consumption(user_id)

# Conversion Analytics Endpoints
@router.get("/analytics/conversion", response_model=ConversionMetrics)
async def get_conversion_metrics(
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """Get conversion rate analytics."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return await analytics_service.get_conversion_metrics()

# Analytics Report Endpoints
@router.get("/reports/{report_type}")
async def get_analytics_report(
    report_type: str,  # daily, weekly, monthly
    start_date: datetime,
    end_date: datetime,
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """Get analytics report for a specific period."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    # Validate report type
    if report_type not in ["daily", "weekly", "monthly"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid report type"
        )
    
    # Validate date range
    if end_date < start_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="End date must be after start date"
        )
    
    # TODO: Implement report generation
    return {
        "report_type": report_type,
        "start_date": start_date,
        "end_date": end_date,
        "status": "generating"
    }

# Feature Usage Tracking
@router.post("/track/feature")
async def track_feature_usage(
    feature_name: str,
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """Track feature usage."""
    await analytics_service.record_feature_usage(current_user.id, feature_name)
    return {"status": "success"}

# Session Tracking
@router.post("/track/session/start")
async def start_session(
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """Start tracking a user session."""
    session_id = await analytics_service.record_user_session(current_user.id)
    return {"session_id": session_id}

@router.post("/track/session/end/{session_id}")
async def end_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """End tracking a user session."""
    await analytics_service.end_user_session(session_id)
    return {"status": "success"} 