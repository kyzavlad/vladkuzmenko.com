import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from functools import lru_cache
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import logging
from datetime import datetime, timedelta
import gc

from .config import settings
from .monitoring import (
    GPU_UTILIZATION,
    MEMORY_USAGE,
    QUEUE_SIZE,
    PROCESSING_TIME
)

class ResourceManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.gpu_memory_limit = settings.GPU_MEMORY_LIMIT * 1024 * 1024 * 1024  # Convert GB to bytes
        self.processing_queue = queue.PriorityQueue()
        self.executor = ThreadPoolExecutor(max_workers=settings.MAX_CONCURRENT_PROCESSING)
        self._lock = threading.Lock()
        self._resource_usage = {}
        self._last_cleanup = datetime.now()
    
    def allocate_resources(self, job_id: str, required_memory: int) -> bool:
        """Allocate resources for a job."""
        with self._lock:
            current_usage = self._get_current_resource_usage()
            if current_usage + required_memory > self.gpu_memory_limit:
                return False
            
            self._resource_usage[job_id] = {
                "memory": required_memory,
                "timestamp": datetime.now()
            }
            return True
    
    def release_resources(self, job_id: str) -> None:
        """Release resources allocated to a job."""
        with self._lock:
            if job_id in self._resource_usage:
                del self._resource_usage[job_id]
    
    def _get_current_resource_usage(self) -> int:
        """Get current GPU memory usage."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated()
        return 0
    
    def cleanup_resources(self) -> None:
        """Clean up unused resources."""
        with self._lock:
            current_time = datetime.now()
            if current_time - self._last_cleanup < timedelta(minutes=5):
                return
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Run garbage collection
            gc.collect()
            
            self._last_cleanup = current_time

class ModelOptimizer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def optimize_model(
        self,
        model: nn.Module,
        optimization_config: Dict[str, Any]
    ) -> nn.Module:
        """Apply optimizations to a model."""
        if optimization_config.get("quantization"):
            model = self._apply_quantization(model)
        
        if optimization_config.get("fuse"):
            model = self._fuse_modules(model)
        
        if optimization_config.get("prune"):
            model = self._prune_model(model)
        
        return model
    
    def _apply_quantization(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization to the model."""
        return torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
    
    def _fuse_modules(self, model: nn.Module) -> nn.Module:
        """Fuse compatible modules for better performance."""
        return torch.quantization.fuse_modules(
            model,
            [['conv', 'bn', 'relu']]
        )
    
    def _prune_model(self, model: nn.Module) -> nn.Module:
        """Apply model pruning."""
        # Implement pruning logic here
        return model

class CacheManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._cache = {}
        self._cache_stats = {}
    
    @lru_cache(maxsize=1000)
    def get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result if available."""
        if cache_key in self._cache:
            self._update_cache_stats(cache_key, hit=True)
            return self._cache[cache_key]
        self._update_cache_stats(cache_key, hit=False)
        return None
    
    def cache_result(self, cache_key: str, result: Any) -> None:
        """Cache a result."""
        self._cache[cache_key] = result
    
    def _update_cache_stats(self, cache_key: str, hit: bool) -> None:
        """Update cache statistics."""
        if cache_key not in self._cache_stats:
            self._cache_stats[cache_key] = {"hits": 0, "misses": 0}
        
        if hit:
            self._cache_stats[cache_key]["hits"] += 1
        else:
            self._cache_stats[cache_key]["misses"] += 1

class BatchProcessor:
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
    
    def process_batch(
        self,
        items: List[Any],
        process_fn: callable,
        **kwargs
    ) -> List[Any]:
        """Process items in batches."""
        results = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = process_fn(batch, **kwargs)
            results.extend(batch_results)
        return results

class PerformanceMonitor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._metrics = {}
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a performance metric."""
        if metric_name not in self._metrics:
            self._metrics[metric_name] = []
        
        self._metrics[metric_name].append({
            "value": value,
            "labels": labels or {},
            "timestamp": datetime.now()
        })
    
    def get_metric_stats(
        self,
        metric_name: str,
        time_window: timedelta = timedelta(hours=1)
    ) -> Dict[str, float]:
        """Get statistics for a metric over a time window."""
        if metric_name not in self._metrics:
            return {}
        
        cutoff_time = datetime.now() - time_window
        recent_metrics = [
            m for m in self._metrics[metric_name]
            if m["timestamp"] > cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        values = [m["value"] for m in recent_metrics]
        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "count": len(values)
        }

class ResourceOptimizer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.resource_manager = ResourceManager()
        self.model_optimizer = ModelOptimizer()
        self.cache_manager = CacheManager()
        self.batch_processor = BatchProcessor()
        self.performance_monitor = PerformanceMonitor()
    
    def optimize_processing(
        self,
        job_id: str,
        input_data: Any,
        process_fn: callable,
        required_memory: int,
        optimization_config: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Optimize processing of a job."""
        # Check cache first
        cache_key = f"{job_id}_{hash(str(input_data))}"
        cached_result = self.cache_manager.get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Allocate resources
        if not self.resource_manager.allocate_resources(job_id, required_memory):
            raise RuntimeError("Insufficient resources available")
        
        try:
            # Process with optimizations
            if optimization_config:
                process_fn = self.model_optimizer.optimize_model(
                    process_fn,
                    optimization_config
                )
            
            # Process in batches if applicable
            if isinstance(input_data, (list, tuple)):
                result = self.batch_processor.process_batch(
                    input_data,
                    process_fn
                )
            else:
                result = process_fn(input_data)
            
            # Cache result
            self.cache_manager.cache_result(cache_key, result)
            
            return result
            
        finally:
            # Release resources
            self.resource_manager.release_resources(job_id)
            # Clean up if needed
            self.resource_manager.cleanup_resources()
    
    def monitor_performance(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        metrics = {
            "gpu_utilization": GPU_UTILIZATION._value.get(),
            "memory_usage": MEMORY_USAGE._value.get(),
            "queue_size": QUEUE_SIZE._value.get(),
            "processing_time": PROCESSING_TIME._value.get()
        }
        
        # Add cache statistics
        metrics["cache_stats"] = self.cache_manager._cache_stats
        
        return metrics 