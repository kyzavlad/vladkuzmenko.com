from typing import Dict, List, Optional
from datetime import datetime, timedelta
import psutil
import time
from prometheus_client import Counter, Gauge, Histogram, Summary
from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from ..database.models import JobDB, TokenUsageDB, TokenTransactionDB
from ..config import settings

# System Metrics
CPU_USAGE = Gauge('system_cpu_usage', 'CPU usage percentage')
MEMORY_USAGE = Gauge('system_memory_usage', 'Memory usage percentage')
DISK_USAGE = Gauge('system_disk_usage', 'Disk usage percentage')
QUEUE_LENGTH = Gauge('job_queue_length', 'Number of jobs in queue')
PROCESSING_BACKLOG = Gauge('job_processing_backlog', 'Number of jobs waiting to be processed')

# Job Metrics
JOB_PROCESSING_TIME = Histogram('job_processing_time', 'Time taken to process jobs')
JOB_ERROR_RATE = Counter('job_error_count', 'Number of job errors')
JOB_SUCCESS_RATE = Counter('job_success_count', 'Number of successful jobs')

# API Metrics
API_REQUEST_LATENCY = Histogram('api_request_latency', 'API request latency')
API_ERROR_RATE = Counter('api_error_count', 'Number of API errors')
API_REQUEST_COUNT = Counter('api_request_count', 'Number of API requests')

# Database Metrics
DB_QUERY_LATENCY = Histogram('db_query_latency', 'Database query latency')
DB_CONNECTION_POOL = Gauge('db_connection_pool', 'Database connection pool size')

# Cache Metrics
CACHE_HIT_RATE = Counter('cache_hit_count', 'Number of cache hits')
CACHE_MISS_RATE = Counter('cache_miss_count', 'Number of cache misses')

class MonitoringService:
    def __init__(self, db: Session):
        self.db = db
        self.start_time = time.time()

    async def collect_system_metrics(self) -> Dict:
        """Collect system performance metrics."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        metrics = {
            'cpu_usage': cpu_percent,
            'memory_usage': memory.percent,
            'disk_usage': disk.percent,
            'uptime': time.time() - self.start_time
        }

        # Update Prometheus metrics
        CPU_USAGE.set(cpu_percent)
        MEMORY_USAGE.set(memory.percent)
        DISK_USAGE.set(disk.percent)

        return metrics

    async def collect_job_metrics(self) -> Dict:
        """Collect job processing metrics."""
        # Get job statistics
        total_jobs = self.db.query(func.count(JobDB.id)).scalar()
        pending_jobs = self.db.query(func.count(JobDB.id))\
            .filter(JobDB.status == 'pending').scalar()
        processing_jobs = self.db.query(func.count(JobDB.id))\
            .filter(JobDB.status == 'processing').scalar()
        completed_jobs = self.db.query(func.count(JobDB.id))\
            .filter(JobDB.status == 'completed').scalar()
        failed_jobs = self.db.query(func.count(JobDB.id))\
            .filter(JobDB.status == 'failed').scalar()

        # Get average processing time
        avg_processing_time = self.db.query(func.avg(JobDB.processing_time))\
            .filter(JobDB.status == 'completed').scalar() or 0

        metrics = {
            'total_jobs': total_jobs,
            'pending_jobs': pending_jobs,
            'processing_jobs': processing_jobs,
            'completed_jobs': completed_jobs,
            'failed_jobs': failed_jobs,
            'average_processing_time': avg_processing_time,
            'error_rate': (failed_jobs / total_jobs * 100) if total_jobs > 0 else 0
        }

        # Update Prometheus metrics
        QUEUE_LENGTH.set(pending_jobs)
        PROCESSING_BACKLOG.set(processing_jobs)

        return metrics

    async def collect_api_metrics(self) -> Dict:
        """Collect API performance metrics."""
        # Get API statistics from the last hour
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        total_requests = self.db.query(func.count(APIRequestDB.id))\
            .filter(APIRequestDB.created_at >= one_hour_ago).scalar()
        error_requests = self.db.query(func.count(APIRequestDB.id))\
            .filter(
                and_(
                    APIRequestDB.created_at >= one_hour_ago,
                    APIRequestDB.status_code >= 400
                )
            ).scalar()

        metrics = {
            'total_requests': total_requests,
            'error_requests': error_requests,
            'error_rate': (error_requests / total_requests * 100) if total_requests > 0 else 0,
            'average_latency': self.db.query(func.avg(APIRequestDB.latency))\
                .filter(APIRequestDB.created_at >= one_hour_ago).scalar() or 0
        }

        return metrics

    async def collect_database_metrics(self) -> Dict:
        """Collect database performance metrics."""
        # Get database statistics
        total_connections = self.db.execute("SELECT count(*) FROM pg_stat_activity").scalar()
        active_connections = self.db.execute(
            "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'"
        ).scalar()

        metrics = {
            'total_connections': total_connections,
            'active_connections': active_connections,
            'connection_pool_size': settings.DB_POOL_SIZE
        }

        # Update Prometheus metrics
        DB_CONNECTION_POOL.set(active_connections)

        return metrics

    async def collect_cache_metrics(self) -> Dict:
        """Collect cache performance metrics."""
        # Get cache statistics
        total_requests = CACHE_HIT_RATE._value.get() + CACHE_MISS_RATE._value.get()
        hit_rate = (CACHE_HIT_RATE._value.get() / total_requests * 100) if total_requests > 0 else 0

        metrics = {
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'miss_rate': 100 - hit_rate
        }

        return metrics

    async def get_all_metrics(self) -> Dict:
        """Collect all system metrics."""
        return {
            'system': await self.collect_system_metrics(),
            'jobs': await self.collect_job_metrics(),
            'api': await self.collect_api_metrics(),
            'database': await self.collect_database_metrics(),
            'cache': await self.collect_cache_metrics(),
            'timestamp': datetime.utcnow().isoformat()
        }

    def record_job_processing_time(self, duration: float):
        """Record job processing time."""
        JOB_PROCESSING_TIME.observe(duration)

    def record_job_error(self):
        """Record job error."""
        JOB_ERROR_RATE.inc()

    def record_job_success(self):
        """Record successful job."""
        JOB_SUCCESS_RATE.inc()

    def record_api_request(self, duration: float, status_code: int):
        """Record API request metrics."""
        API_REQUEST_LATENCY.observe(duration)
        API_REQUEST_COUNT.inc()
        if status_code >= 400:
            API_ERROR_RATE.inc()

    def record_db_query(self, duration: float):
        """Record database query latency."""
        DB_QUERY_LATENCY.observe(duration)

    def record_cache_hit(self):
        """Record cache hit."""
        CACHE_HIT_RATE.inc()

    def record_cache_miss(self):
        """Record cache miss."""
        CACHE_MISS_RATE.inc() 