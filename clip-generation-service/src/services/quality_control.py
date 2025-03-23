from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from ..database.models import (
    JobDB, QualityMetricsDB, ABTestDB,
    UserFeedbackDB, QualityReportDB
)
from ..models.analytics import QualityMetrics

class QualityControlService:
    def __init__(self, db: Session):
        self.db = db

    async def analyze_output_quality(self, job_id: str) -> QualityMetrics:
        """Analyze the quality of a processed video output."""
        job = self.db.query(JobDB).filter(JobDB.id == job_id).first()
        if not job:
            raise ValueError(f"Job {job_id} not found")

        # Get video and audio metrics
        video_metrics = await self._analyze_video_quality(job.output_path)
        audio_metrics = await self._analyze_audio_quality(job.output_path)
        
        # Calculate overall quality score
        overall_score = self._calculate_overall_score(video_metrics, audio_metrics)
        
        # Store quality metrics
        metrics = QualityMetricsDB(
            id=str(uuid4()),
            job_id=job_id,
            video_metrics=video_metrics,
            audio_metrics=audio_metrics,
            overall_score=overall_score,
            created_at=datetime.utcnow()
        )
        self.db.add(metrics)
        self.db.commit()
        
        return QualityMetrics(
            video_metrics=video_metrics,
            audio_metrics=audio_metrics,
            overall_score=overall_score
        )

    async def _analyze_video_quality(self, video_path: str) -> Dict[str, float]:
        """Analyze video quality metrics."""
        # Implement video quality analysis
        # This would use FFmpeg and other tools to analyze:
        # - Resolution and aspect ratio
        # - Frame rate consistency
        # - Bitrate and compression quality
        # - Color accuracy
        # - Motion smoothness
        # - Artifact detection
        pass

    async def _analyze_audio_quality(self, video_path: str) -> Dict[str, float]:
        """Analyze audio quality metrics."""
        # Implement audio quality analysis
        # This would analyze:
        # - Audio levels and normalization
        # - Frequency response
        # - Dynamic range
        # - Clipping and distortion
        # - Background noise
        pass

    def _calculate_overall_score(
        self,
        video_metrics: Dict[str, float],
        audio_metrics: Dict[str, float]
    ) -> float:
        """Calculate overall quality score from metrics."""
        # Weight different metrics based on importance
        weights = {
            "video": {
                "resolution": 0.2,
                "frame_rate": 0.15,
                "bitrate": 0.15,
                "color": 0.1,
                "motion": 0.1,
                "artifacts": 0.2
            },
            "audio": {
                "levels": 0.3,
                "frequency": 0.2,
                "dynamic_range": 0.2,
                "distortion": 0.15,
                "noise": 0.15
            }
        }
        
        # Calculate weighted scores
        video_score = sum(
            video_metrics[metric] * weight
            for metric, weight in weights["video"].items()
        )
        audio_score = sum(
            audio_metrics[metric] * weight
            for metric, weight in weights["audio"].items()
        )
        
        # Combine scores with 70% video, 30% audio weighting
        return 0.7 * video_score + 0.3 * audio_score

    async def setup_ab_test(
        self,
        job_id: str,
        variant_a: Dict[str, any],
        variant_b: Dict[str, any]
    ) -> str:
        """Set up an A/B test for different processing parameters."""
        test = ABTestDB(
            id=str(uuid4()),
            job_id=job_id,
            variant_a=variant_a,
            variant_b=variant_b,
            status="active",
            created_at=datetime.utcnow()
        )
        self.db.add(test)
        self.db.commit()
        return test.id

    async def record_ab_test_result(
        self,
        test_id: str,
        variant: str,
        metrics: QualityMetrics
    ):
        """Record results for an A/B test variant."""
        test = self.db.query(ABTestDB).filter(ABTestDB.id == test_id).first()
        if not test:
            raise ValueError(f"Test {test_id} not found")
        
        if variant == "a":
            test.variant_a_metrics = metrics.dict()
        else:
            test.variant_b_metrics = metrics.dict()
        
        # Check if both variants have results
        if test.variant_a_metrics and test.variant_b_metrics:
            test.status = "completed"
            test.completed_at = datetime.utcnow()
        
        self.db.commit()

    async def analyze_ab_test_results(self, test_id: str) -> Dict:
        """Analyze results of an A/B test."""
        test = self.db.query(ABTestDB).filter(ABTestDB.id == test_id).first()
        if not test or test.status != "completed":
            raise ValueError(f"Test {test_id} not completed")
        
        # Compare metrics between variants
        comparison = {
            "overall_score": {
                "variant_a": test.variant_a_metrics["overall_score"],
                "variant_b": test.variant_b_metrics["overall_score"],
                "difference": (
                    test.variant_b_metrics["overall_score"] -
                    test.variant_a_metrics["overall_score"]
                )
            },
            "video_metrics": {},
            "audio_metrics": {}
        }
        
        # Compare individual metrics
        for metric in test.variant_a_metrics["video_metrics"]:
            comparison["video_metrics"][metric] = {
                "variant_a": test.variant_a_metrics["video_metrics"][metric],
                "variant_b": test.variant_b_metrics["video_metrics"][metric],
                "difference": (
                    test.variant_b_metrics["video_metrics"][metric] -
                    test.variant_a_metrics["video_metrics"][metric]
                )
            }
        
        for metric in test.variant_a_metrics["audio_metrics"]:
            comparison["audio_metrics"][metric] = {
                "variant_a": test.variant_a_metrics["audio_metrics"][metric],
                "variant_b": test.variant_b_metrics["audio_metrics"][metric],
                "difference": (
                    test.variant_b_metrics["audio_metrics"][metric] -
                    test.variant_a_metrics["audio_metrics"][metric]
                )
            }
        
        return comparison

    async def collect_user_feedback(
        self,
        job_id: str,
        user_id: str,
        rating: int,
        feedback: str
    ):
        """Collect user feedback for a processed video."""
        feedback_record = UserFeedbackDB(
            id=str(uuid4()),
            job_id=job_id,
            user_id=user_id,
            rating=rating,
            feedback=feedback,
            created_at=datetime.utcnow()
        )
        self.db.add(feedback_record)
        self.db.commit()

    async def generate_quality_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """Generate a quality report for a time period."""
        # Get all jobs in the period
        jobs = self.db.query(JobDB)\
            .filter(
                and_(
                    JobDB.created_at >= start_date,
                    JobDB.created_at <= end_date,
                    JobDB.status == "completed"
                )
            ).all()
        
        # Collect quality metrics
        quality_metrics = []
        for job in jobs:
            metrics = self.db.query(QualityMetricsDB)\
                .filter(QualityMetricsDB.job_id == job.id)\
                .first()
            if metrics:
                quality_metrics.append(metrics)
        
        # Calculate statistics
        overall_scores = [m.overall_score for m in quality_metrics]
        report = {
            "period": {
                "start": start_date,
                "end": end_date
            },
            "total_jobs": len(jobs),
            "quality_metrics": {
                "average_score": statistics.mean(overall_scores),
                "median_score": statistics.median(overall_scores),
                "std_dev": statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0,
                "min_score": min(overall_scores),
                "max_score": max(overall_scores)
            },
            "video_metrics": self._aggregate_metrics(
                [m.video_metrics for m in quality_metrics]
            ),
            "audio_metrics": self._aggregate_metrics(
                [m.audio_metrics for m in quality_metrics]
            )
        }
        
        # Store report
        report_record = QualityReportDB(
            id=str(uuid4()),
            start_date=start_date,
            end_date=end_date,
            metrics=report,
            created_at=datetime.utcnow()
        )
        self.db.add(report_record)
        self.db.commit()
        
        return report

    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict:
        """Aggregate metrics across multiple samples."""
        if not metrics_list:
            return {}
        
        aggregated = {}
        for metric in metrics_list[0].keys():
            values = [m[metric] for m in metrics_list]
            aggregated[metric] = {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                "min": min(values),
                "max": max(values)
            }
        return aggregated

    async def detect_quality_degradation(
        self,
        current_metrics: QualityMetrics,
        threshold: float = 0.1
    ) -> bool:
        """Detect significant quality degradation."""
        # Get recent quality metrics for comparison
        recent_metrics = self.db.query(QualityMetricsDB)\
            .order_by(QualityMetricsDB.created_at.desc())\
            .limit(100)\
            .all()
        
        if not recent_metrics:
            return False
        
        # Calculate average recent quality
        recent_scores = [m.overall_score for m in recent_metrics]
        avg_recent_score = statistics.mean(recent_scores)
        
        # Check for significant degradation
        degradation = avg_recent_score - current_metrics.overall_score
        return degradation > threshold 