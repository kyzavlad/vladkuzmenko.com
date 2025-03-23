from prometheus_client import Counter, Histogram, Gauge
from prometheus_fastapi_instrumentator import Instrumentator
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

from .config import settings

# Prometheus metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"]
)

PROCESSING_TIME = Histogram(
    "video_processing_duration_seconds",
    "Video processing duration in seconds",
    ["operation_type"]
)

QUEUE_SIZE = Gauge(
    "processing_queue_size",
    "Current size of the processing queue",
    ["queue_type"]
)

GPU_UTILIZATION = Gauge(
    "gpu_utilization_percent",
    "GPU utilization percentage",
    ["device_id"]
)

MEMORY_USAGE = Gauge(
    "memory_usage_bytes",
    "Memory usage in bytes",
    ["component"]
)

# OpenTelemetry metrics
def setup_metrics():
    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(
            endpoint="http://localhost:4317",
            insecure=True
        )
    )
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)
    return metrics.get_meter(__name__)

# Prometheus instrumentation
def setup_prometheus(app):
    Instrumentator().instrument(app).expose(app)

# Custom metrics
def record_request_metrics(method: str, endpoint: str, status: int, duration: float):
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)

def record_processing_metrics(operation_type: str, duration: float):
    PROCESSING_TIME.labels(operation_type=operation_type).observe(duration)

def update_queue_metrics(queue_type: str, size: int):
    QUEUE_SIZE.labels(queue_type=queue_type).set(size)

def update_resource_metrics(device_id: str, gpu_util: float, memory_used: int):
    GPU_UTILIZATION.labels(device_id=device_id).set(gpu_util)
    MEMORY_USAGE.labels(component="gpu").set(memory_used) 