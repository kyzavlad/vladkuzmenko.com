import logging
import sys
from logging.handlers import RotatingFileHandler
from elasticsearch import AsyncElasticsearch
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from datetime import datetime

from .config import settings

# Configure logging
def setup_logging():
    logger = logging.getLogger("ai_video_platform")
    logger.setLevel(settings.LOG_LEVEL)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(settings.LOG_LEVEL)
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = RotatingFileHandler(
        "logs/app.log",
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(settings.LOG_LEVEL)
    file_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # Elasticsearch handler
    es_client = AsyncElasticsearch([settings.ELASTICSEARCH_URL])
    es_handler = ElasticsearchHandler(es_client)
    es_handler.setLevel(settings.LOG_LEVEL)
    es_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    es_handler.setFormatter(es_format)
    logger.addHandler(es_handler)
    
    return logger

# Configure tracing
def setup_tracing():
    tracer_provider = TracerProvider()
    processor = BatchSpanProcessor(
        OTLPSpanExporter(
            endpoint="http://localhost:4317",
            insecure=True
        )
    )
    tracer_provider.add_span_processor(processor)
    trace.set_tracer_provider(tracer_provider)
    return trace.get_tracer(__name__)

# Elasticsearch handler
class ElasticsearchHandler(logging.Handler):
    def __init__(self, client):
        super().__init__()
        self.client = client
    
    def emit(self, record):
        try:
            log_entry = {
                "timestamp": record.created,
                "level": record.levelname,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
                "extra": getattr(record, "extra", {})
            }
            self.client.index(
                index=f"logs-{datetime.now().strftime('%Y.%m.%d')}",
                document=log_entry
            )
        except Exception:
            self.handleError(record) 