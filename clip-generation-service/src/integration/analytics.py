import os
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
from datetime import datetime
import asyncio
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

@dataclass
class AnalyticsConfig:
    """Configuration for analytics extensions."""
    analytics_dir: str = "analytics"
    clip_performance_dir: str = "clip_performance"
    ab_test_dir: str = "ab_tests"
    content_classification_dir: str = "content_classification"
    engagement_metrics_dir: str = "engagement_metrics"
    min_samples_for_ab_test: int = 100
    confidence_threshold: float = 0.95

class AnalyticsIntegrator:
    """Integrates analytics extensions for clip performance tracking."""
    
    def __init__(self, config: AnalyticsConfig):
        """
        Initialize analytics integrator.
        
        Args:
            config (AnalyticsConfig): Analytics configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Create analytics directories
        for dir_name in [
            config.analytics_dir,
            config.clip_performance_dir,
            config.ab_test_dir,
            config.content_classification_dir,
            config.engagement_metrics_dir
        ]:
            os.makedirs(dir_name, exist_ok=True)
    
    async def track_clip_performance(
        self,
        clip_id: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict] = None
    ):
        """
        Track performance metrics for a clip.
        
        Args:
            clip_id (str): Clip identifier
            metrics (Dict[str, float]): Performance metrics
            metadata (Optional[Dict]): Additional metadata
        """
        # Create performance record
        record = {
            "clip_id": clip_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "metadata": metadata or {}
        }
        
        # Save to analytics
        await self._save_clip_performance(record)
    
    async def _save_clip_performance(self, record: Dict):
        """
        Save clip performance record.
        
        Args:
            record (Dict): Performance record
        """
        # Create filename with date
        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"clip_performance_{date_str}.json"
        filepath = os.path.join(self.config.clip_performance_dir, filename)
        
        # Load existing records
        records = []
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                records = json.load(f)
        
        # Add new record
        records.append(record)
        
        # Save updated records
        with open(filepath, "w") as f:
            json.dump(records, f, indent=2)
    
    async def run_ab_test(
        self,
        test_id: str,
        variants: List[Dict],
        metrics: List[str],
        duration_days: int = 7
    ) -> Dict:
        """
        Run A/B test for clip variants.
        
        Args:
            test_id (str): Test identifier
            variants (List[Dict]): Clip variants
            metrics (List[str]): Metrics to track
            duration_days (int): Test duration in days
            
        Returns:
            Dict: Test results
        """
        # Create test record
        test_record = {
            "test_id": test_id,
            "start_time": datetime.now().isoformat(),
            "end_time": (datetime.now() + pd.Timedelta(days=duration_days)).isoformat(),
            "variants": variants,
            "metrics": metrics,
            "status": "running"
        }
        
        # Save test configuration
        test_path = os.path.join(
            self.config.ab_test_dir,
            f"test_{test_id}.json"
        )
        
        with open(test_path, "w") as f:
            json.dump(test_record, f, indent=2)
        
        # Track test results
        results = await self._track_ab_test_results(test_id, metrics)
        
        # Analyze results
        analysis = self._analyze_ab_test_results(results)
        
        # Update test record
        test_record.update({
            "status": "completed",
            "results": results,
            "analysis": analysis
        })
        
        with open(test_path, "w") as f:
            json.dump(test_record, f, indent=2)
        
        return test_record
    
    async def _track_ab_test_results(
        self,
        test_id: str,
        metrics: List[str]
    ) -> Dict[str, List[Dict]]:
        """
        Track results for A/B test.
        
        Args:
            test_id (str): Test identifier
            metrics (List[str]): Metrics to track
            
        Returns:
            Dict[str, List[Dict]]: Test results
        """
        results = {metric: [] for metric in metrics}
        
        # Load performance records
        for filename in os.listdir(self.config.clip_performance_dir):
            if not filename.startswith("clip_performance_"):
                continue
            
            filepath = os.path.join(self.config.clip_performance_dir, filename)
            with open(filepath, "r") as f:
                records = json.load(f)
                
                for record in records:
                    if record["metadata"].get("test_id") == test_id:
                        for metric in metrics:
                            if metric in record["metrics"]:
                                results[metric].append({
                                    "variant": record["metadata"]["variant"],
                                    "value": record["metrics"][metric],
                                    "timestamp": record["timestamp"]
                                })
        
        return results
    
    def _analyze_ab_test_results(
        self,
        results: Dict[str, List[Dict]]
    ) -> Dict[str, Dict]:
        """
        Analyze A/B test results.
        
        Args:
            results (Dict[str, List[Dict]]): Test results
            
        Returns:
            Dict[str, Dict]: Analysis results
        """
        analysis = {}
        
        for metric, metric_results in results.items():
            # Convert to DataFrame
            df = pd.DataFrame(metric_results)
            
            # Group by variant
            variant_stats = df.groupby("variant")["value"].agg([
                "count",
                "mean",
                "std",
                "min",
                "max"
            ]).to_dict()
            
            # Perform statistical test
            from scipy import stats
            variants = df["variant"].unique()
            if len(variants) == 2:
                v1_data = df[df["variant"] == variants[0]]["value"]
                v2_data = df[df["variant"] == variants[1]]["value"]
                
                t_stat, p_value = stats.ttest_ind(v1_data, v2_data)
                
                analysis[metric] = {
                    "variant_stats": variant_stats,
                    "statistical_test": {
                        "t_statistic": t_stat,
                        "p_value": p_value,
                        "significant": p_value < 0.05
                    }
                }
            else:
                analysis[metric] = {
                    "variant_stats": variant_stats,
                    "statistical_test": None
                }
        
        return analysis
    
    async def track_engagement(
        self,
        clip_id: str,
        metrics: Dict[str, float],
        user_data: Optional[Dict] = None
    ):
        """
        Track user engagement metrics for a clip.
        
        Args:
            clip_id (str): Clip identifier
            metrics (Dict[str, float]): Engagement metrics
            user_data (Optional[Dict]): User data
        """
        # Create engagement record
        record = {
            "clip_id": clip_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "user_data": user_data or {}
        }
        
        # Save to analytics
        await self._save_engagement_metrics(record)
    
    async def _save_engagement_metrics(self, record: Dict):
        """
        Save engagement metrics record.
        
        Args:
            record (Dict): Engagement record
        """
        # Create filename with date
        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"engagement_metrics_{date_str}.json"
        filepath = os.path.join(self.config.engagement_metrics_dir, filename)
        
        # Load existing records
        records = []
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                records = json.load(f)
        
        # Add new record
        records.append(record)
        
        # Save updated records
        with open(filepath, "w") as f:
            json.dump(records, f, indent=2)
    
    async def classify_content(
        self,
        clip_id: str,
        content_data: Dict,
        num_clusters: int = 5
    ) -> Dict:
        """
        Classify clip content.
        
        Args:
            clip_id (str): Clip identifier
            content_data (Dict): Content data
            num_clusters (int): Number of clusters for topic modeling
            
        Returns:
            Dict: Classification results
        """
        # Extract features
        features = self._extract_content_features(content_data)
        
        # Perform topic modeling
        topics = self._perform_topic_modeling(features, num_clusters)
        
        # Create classification record
        record = {
            "clip_id": clip_id,
            "timestamp": datetime.now().isoformat(),
            "features": features,
            "topics": topics,
            "content_data": content_data
        }
        
        # Save to analytics
        await self._save_content_classification(record)
        
        return record
    
    def _extract_content_features(self, content_data: Dict) -> Dict:
        """
        Extract features from content data.
        
        Args:
            content_data (Dict): Content data
            
        Returns:
            Dict: Extracted features
        """
        features = {}
        
        # Extract text features
        if "transcription" in content_data:
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words="english"
            )
            tfidf_matrix = vectorizer.fit_transform([content_data["transcription"]])
            features["text_features"] = tfidf_matrix.toarray()[0].tolist()
        
        # Extract visual features
        if "visual_features" in content_data:
            features["visual_features"] = content_data["visual_features"]
        
        # Extract audio features
        if "audio_features" in content_data:
            features["audio_features"] = content_data["audio_features"]
        
        return features
    
    def _perform_topic_modeling(
        self,
        features: Dict,
        num_clusters: int
    ) -> Dict:
        """
        Perform topic modeling on features.
        
        Args:
            features (Dict): Content features
            num_clusters (int): Number of clusters
            
        Returns:
            Dict: Topic modeling results
        """
        results = {}
        
        # Combine features for clustering
        if "text_features" in features:
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            clusters = kmeans.fit_predict(features["text_features"])
            
            # Get cluster centers
            centers = kmeans.cluster_centers_
            
            results["text_topics"] = {
                "cluster_labels": clusters.tolist(),
                "cluster_centers": centers.tolist()
            }
        
        return results
    
    async def _save_content_classification(self, record: Dict):
        """
        Save content classification record.
        
        Args:
            record (Dict): Classification record
        """
        # Create filename with date
        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"content_classification_{date_str}.json"
        filepath = os.path.join(self.config.content_classification_dir, filename)
        
        # Load existing records
        records = []
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                records = json.load(f)
        
        # Add new record
        records.append(record)
        
        # Save updated records
        with open(filepath, "w") as f:
            json.dump(records, f, indent=2)
    
    async def generate_analytics_report(
        self,
        clip_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """
        Generate comprehensive analytics report for a clip.
        
        Args:
            clip_id (str): Clip identifier
            start_date (Optional[datetime]): Start date
            end_date (Optional[datetime]): End date
            
        Returns:
            Dict: Analytics report
        """
        # Load performance data
        performance_data = await self._load_clip_performance(
            clip_id,
            start_date,
            end_date
        )
        
        # Load engagement data
        engagement_data = await self._load_engagement_metrics(
            clip_id,
            start_date,
            end_date
        )
        
        # Load classification data
        classification_data = await self._load_content_classification(clip_id)
        
        # Generate report
        report = {
            "clip_id": clip_id,
            "period": {
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None
            },
            "performance": self._analyze_performance(performance_data),
            "engagement": self._analyze_engagement(engagement_data),
            "classification": classification_data
        }
        
        return report
    
    async def _load_clip_performance(
        self,
        clip_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Load clip performance data.
        
        Args:
            clip_id (str): Clip identifier
            start_date (Optional[datetime]): Start date
            end_date (Optional[datetime]): End date
            
        Returns:
            List[Dict]: Performance data
        """
        records = []
        
        # Get date range
        if start_date is None:
            start_date = datetime.now().replace(day=1)  # First day of current month
        if end_date is None:
            end_date = datetime.now()
        
        # Load records from performance directory
        for filename in os.listdir(self.config.clip_performance_dir):
            if not filename.startswith("clip_performance_"):
                continue
            
            filepath = os.path.join(self.config.clip_performance_dir, filename)
            with open(filepath, "r") as f:
                file_records = json.load(f)
                
                for record in file_records:
                    if record["clip_id"] != clip_id:
                        continue
                    
                    record_date = datetime.fromisoformat(record["timestamp"])
                    if start_date <= record_date <= end_date:
                        records.append(record)
        
        return records
    
    async def _load_engagement_metrics(
        self,
        clip_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Load engagement metrics data.
        
        Args:
            clip_id (str): Clip identifier
            start_date (Optional[datetime]): Start date
            end_date (Optional[datetime]): End date
            
        Returns:
            List[Dict]: Engagement data
        """
        records = []
        
        # Get date range
        if start_date is None:
            start_date = datetime.now().replace(day=1)  # First day of current month
        if end_date is None:
            end_date = datetime.now()
        
        # Load records from engagement directory
        for filename in os.listdir(self.config.engagement_metrics_dir):
            if not filename.startswith("engagement_metrics_"):
                continue
            
            filepath = os.path.join(self.config.engagement_metrics_dir, filename)
            with open(filepath, "r") as f:
                file_records = json.load(f)
                
                for record in file_records:
                    if record["clip_id"] != clip_id:
                        continue
                    
                    record_date = datetime.fromisoformat(record["timestamp"])
                    if start_date <= record_date <= end_date:
                        records.append(record)
        
        return records
    
    async def _load_content_classification(
        self,
        clip_id: str
    ) -> Optional[Dict]:
        """
        Load content classification data.
        
        Args:
            clip_id (str): Clip identifier
            
        Returns:
            Optional[Dict]: Classification data
        """
        # Load records from classification directory
        for filename in os.listdir(self.config.content_classification_dir):
            if not filename.startswith("content_classification_"):
                continue
            
            filepath = os.path.join(self.config.content_classification_dir, filename)
            with open(filepath, "r") as f:
                records = json.load(f)
                
                for record in records:
                    if record["clip_id"] == clip_id:
                        return record
        
        return None
    
    def _analyze_performance(self, data: List[Dict]) -> Dict:
        """
        Analyze performance data.
        
        Args:
            data (List[Dict]): Performance data
            
        Returns:
            Dict: Performance analysis
        """
        if not data:
            return {}
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Calculate metrics
        metrics = {}
        for column in df.columns:
            if column not in ["clip_id", "timestamp", "metadata"]:
                metrics[column] = {
                    "mean": float(df[column].mean()),
                    "std": float(df[column].std()),
                    "min": float(df[column].min()),
                    "max": float(df[column].max()),
                    "count": int(df[column].count())
                }
        
        return metrics
    
    def _analyze_engagement(self, data: List[Dict]) -> Dict:
        """
        Analyze engagement data.
        
        Args:
            data (List[Dict]): Engagement data
            
        Returns:
            Dict: Engagement analysis
        """
        if not data:
            return {}
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Calculate metrics
        metrics = {}
        for column in df.columns:
            if column not in ["clip_id", "timestamp", "user_data"]:
                metrics[column] = {
                    "mean": float(df[column].mean()),
                    "std": float(df[column].std()),
                    "min": float(df[column].min()),
                    "max": float(df[column].max()),
                    "count": int(df[column].count())
                }
        
        # Analyze user demographics if available
        if "user_data" in df.columns:
            user_data = pd.DataFrame(df["user_data"].tolist())
            demographics = {}
            for column in user_data.columns:
                demographics[column] = user_data[column].value_counts().to_dict()
            metrics["demographics"] = demographics
        
        return metrics

def main():
    """Main function for analytics integration."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_id", type=str, required=True)
    parser.add_argument("--start_date", type=str, default=None)
    parser.add_argument("--end_date", type=str, default=None)
    parser.add_argument("--metrics", nargs="+", default=[])
    parser.add_argument("--user_data", type=str, default=None)
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create analytics configuration
    config = AnalyticsConfig()
    
    # Create integrator
    integrator = AnalyticsIntegrator(config)
    
    # Track performance
    if args.metrics:
        metrics = {metric: 0.0 for metric in args.metrics}  # Example metrics
        user_data = json.loads(args.user_data) if args.user_data else None
        
        asyncio.run(integrator.track_clip_performance(
            args.clip_id,
            metrics,
            {"user_data": user_data} if user_data else None
        ))
    
    # Generate report
    start_date = datetime.fromisoformat(args.start_date) if args.start_date else None
    end_date = datetime.fromisoformat(args.end_date) if args.end_date else None
    
    report = asyncio.run(integrator.generate_analytics_report(
        args.clip_id,
        start_date,
        end_date
    ))
    
    print("\nAnalytics Report:")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main() 