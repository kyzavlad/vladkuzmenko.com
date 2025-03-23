from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentation
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from datetime import datetime
import asyncio

from .api.v1 import (
    auth,
    video_editing,
    avatar,
    translation,
    analytics,
    billing,
)
from .core.config import settings
from .core.logging import setup_logging
from .core.middleware import RequestLoggingMiddleware
from .core.monitoring import setup_monitoring
from .core.security import setup_security
from .db.session import setup_database
from .core.optimization import ResourceOptimizer

# Initialize metrics
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

# Setup logging
logger = setup_logging()

# Initialize resource optimizer
resource_optimizer = ResourceOptimizer()

app = FastAPI(
    title="AI Video Platform",
    description="Advanced AI-powered video editing and processing platform",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Setup middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)

# Setup monitoring
Instrumentator().instrument(app).expose(app)
FastAPIInstrumentation().instrument(app)

# Setup security
setup_security(app)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(video_editing.router, prefix="/api/video", tags=["Video Editing"])
app.include_router(avatar.router, prefix="/api/avatar", tags=["Avatar"])
app.include_router(translation.router, prefix="/api/translation", tags=["Translation"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["Analytics"])
app.include_router(billing.router, prefix="/api/billing", tags=["Billing"])

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting up AI Video Platform...")
    await setup_database()
    await setup_monitoring()
    logger.info("AI Video Platform startup complete!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down AI Video Platform...")
    # Add cleanup code here
    logger.info("AI Video Platform shutdown complete!")

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/metrics")
async def get_metrics():
    """Get current performance metrics."""
    return resource_optimizer.monitor_performance()

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Global error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "request_id": request.state.request_id
        }
    )

# Background task for resource cleanup
async def cleanup_resources():
    """Periodic resource cleanup task."""
    while True:
        try:
            resource_optimizer.resource_manager.cleanup_resources()
            await asyncio.sleep(300)  # Run every 5 minutes
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
            await asyncio.sleep(60)  # Wait a minute before retrying

@app.on_event("startup")
async def start_cleanup_task():
    """Start the background cleanup task."""
    asyncio.create_task(cleanup_resources()) 