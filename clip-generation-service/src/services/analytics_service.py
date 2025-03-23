from typing import Dict, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, desc
from uuid import uuid4

from ..database.models import (
    JobDB, TokenUsageDB, TokenTransactionDB, UserDB,
    UserSessionDB, FeatureUsageDB
)
from ..models.analytics import (
    UserAnalytics, FeatureUsage, UserEngagement,
    TokenConsumption, ConversionMetrics
)

class AnalyticsService:
    def __init__(self, db: Session):
        self.db = db

    async def get_user_analytics(self, user_id: str) -> UserAnalytics:
        """Get comprehensive analytics for a user."""
        # Get basic usage statistics
        total_jobs = self.db.query(func.count(JobDB.id))\
            .filter(JobDB.user_id == user_id).scalar()
        completed_jobs = self.db.query(func.count(JobDB.id))\
            .filter(
                and_(
                    JobDB.user_id == user_id,
                    JobDB.status == 'completed'
                )
            ).scalar()

        # Get token consumption
        token_usage = self.db.query(
            func.sum(TokenUsageDB.tokens_used).label('total_tokens'),
            func.avg(TokenUsageDB.tokens_used).label('avg_tokens'),
            func.sum(TokenUsageDB.duration).label('total_duration')
        ).filter(TokenUsageDB.user_id == user_id).first()

        # Get feature usage
        feature_usage = self.db.query(
            FeatureUsageDB.feature_name,
            func.count(FeatureUsageDB.id).label('usage_count')
        ).filter(
            FeatureUsageDB.user_id == user_id
        ).group_by(
            FeatureUsageDB.feature_name
        ).order_by(
            desc('usage_count')
        ).limit(5).all()

        # Get session data
        sessions = self.db.query(UserSessionDB)\
            .filter(UserSessionDB.user_id == user_id)\
            .order_by(UserSessionDB.created_at.desc())\
            .limit(10).all()

        return UserAnalytics(
            user_id=user_id,
            total_jobs=total_jobs,
            completed_jobs=completed_jobs,
            success_rate=(completed_jobs / total_jobs * 100) if total_jobs > 0 else 0,
            total_tokens_used=token_usage.total_tokens or 0,
            average_tokens_per_job=token_usage.avg_tokens or 0,
            total_processing_time=token_usage.total_duration or 0,
            most_used_features=[f.feature_name for f in feature_usage],
            recent_sessions=[{
                'start_time': s.created_at,
                'duration': (s.end_time - s.created_at).total_seconds() if s.end_time else None
            } for s in sessions]
        )

    async def get_feature_usage(self, feature_name: str) -> FeatureUsage:
        """Get usage statistics for a specific feature."""
        # Get total usage
        total_usage = self.db.query(func.count(FeatureUsageDB.id))\
            .filter(FeatureUsageDB.feature_name == feature_name).scalar()

        # Get usage by time period
        now = datetime.utcnow()
        last_24h = now - timedelta(days=1)
        last_7d = now - timedelta(days=7)
        last_30d = now - timedelta(days=30)

        usage_24h = self.db.query(func.count(FeatureUsageDB.id))\
            .filter(
                and_(
                    FeatureUsageDB.feature_name == feature_name,
                    FeatureUsageDB.created_at >= last_24h
                )
            ).scalar()

        usage_7d = self.db.query(func.count(FeatureUsageDB.id))\
            .filter(
                and_(
                    FeatureUsageDB.feature_name == feature_name,
                    FeatureUsageDB.created_at >= last_7d
                )
            ).scalar()

        usage_30d = self.db.query(func.count(FeatureUsageDB.id))\
            .filter(
                and_(
                    FeatureUsageDB.feature_name == feature_name,
                    FeatureUsageDB.created_at >= last_30d
                )
            ).scalar()

        return FeatureUsage(
            feature_name=feature_name,
            total_usage=total_usage,
            usage_24h=usage_24h,
            usage_7d=usage_7d,
            usage_30d=usage_30d,
            growth_rate=((usage_24h / usage_7d * 100) - 100) if usage_7d > 0 else 0
        )

    async def get_user_engagement(self, user_id: str) -> UserEngagement:
        """Get user engagement metrics."""
        # Get session data
        sessions = self.db.query(UserSessionDB)\
            .filter(UserSessionDB.user_id == user_id)\
            .order_by(UserSessionDB.created_at.desc()).all()

        # Calculate engagement metrics
        total_sessions = len(sessions)
        total_duration = sum(
            (s.end_time - s.created_at).total_seconds()
            for s in sessions if s.end_time
        )
        avg_session_duration = total_duration / total_sessions if total_sessions > 0 else 0

        # Get feature interaction
        feature_interactions = self.db.query(
            FeatureUsageDB.feature_name,
            func.count(FeatureUsageDB.id).label('interaction_count')
        ).filter(
            FeatureUsageDB.user_id == user_id
        ).group_by(
            FeatureUsageDB.feature_name
        ).all()

        return UserEngagement(
            user_id=user_id,
            total_sessions=total_sessions,
            total_duration=total_duration,
            average_session_duration=avg_session_duration,
            feature_interactions={
                f.feature_name: f.interaction_count
                for f in feature_interactions
            }
        )

    async def get_token_consumption(self, user_id: str) -> TokenConsumption:
        """Get token consumption patterns."""
        # Get daily token usage
        daily_usage = self.db.query(
            func.date(TokenUsageDB.created_at).label('date'),
            func.sum(TokenUsageDB.tokens_used).label('total_tokens')
        ).filter(
            TokenUsageDB.user_id == user_id
        ).group_by(
            func.date(TokenUsageDB.created_at)
        ).order_by(
            desc('date')
        ).limit(30).all()

        # Get feature-specific token usage
        feature_usage = self.db.query(
            JobDB.feature,
            func.sum(TokenUsageDB.tokens_used).label('total_tokens')
        ).join(
            TokenUsageDB, JobDB.id == TokenUsageDB.job_id
        ).filter(
            TokenUsageDB.user_id == user_id
        ).group_by(
            JobDB.feature
        ).all()

        return TokenConsumption(
            user_id=user_id,
            daily_usage={
                str(usage.date): usage.total_tokens
                for usage in daily_usage
            },
            feature_usage={
                usage.feature: usage.total_tokens
                for usage in feature_usage
            }
        )

    async def get_conversion_metrics(self) -> ConversionMetrics:
        """Get conversion rate analytics."""
        # Get user registration data
        total_users = self.db.query(func.count(UserDB.id)).scalar()
        active_users = self.db.query(func.count(UserDB.id))\
            .filter(UserDB.is_active == True).scalar()

        # Get subscription data
        subscribed_users = self.db.query(func.count(UserDB.id))\
            .filter(UserDB.has_subscription == True).scalar()

        # Get feature adoption
        feature_adoption = self.db.query(
            FeatureUsageDB.feature_name,
            func.count(func.distinct(FeatureUsageDB.user_id)).label('user_count')
        ).group_by(
            FeatureUsageDB.feature_name
        ).all()

        return ConversionMetrics(
            total_users=total_users,
            active_users=active_users,
            subscription_rate=(subscribed_users / total_users * 100) if total_users > 0 else 0,
            feature_adoption={
                f.feature_name: f.user_count
                for f in feature_adoption
            }
        )

    async def record_feature_usage(self, user_id: str, feature_name: str):
        """Record feature usage."""
        usage = FeatureUsageDB(
            id=str(uuid4()),
            user_id=user_id,
            feature_name=feature_name,
            created_at=datetime.utcnow()
        )
        self.db.add(usage)
        self.db.commit()

    async def record_user_session(self, user_id: str):
        """Record user session start."""
        session = UserSessionDB(
            id=str(uuid4()),
            user_id=user_id,
            created_at=datetime.utcnow()
        )
        self.db.add(session)
        self.db.commit()
        return session.id

    async def end_user_session(self, session_id: str):
        """Record user session end."""
        session = self.db.query(UserSessionDB)\
            .filter(UserSessionDB.id == session_id).first()
        if session:
            session.end_time = datetime.utcnow()
            self.db.commit() 