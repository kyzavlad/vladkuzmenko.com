import logging
import os
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import time

from app.core.config import settings
from app.db.session import init_db
from app.api.transcriptions import router as transcriptions_router
from app.services.auth import verify_service_api_key

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)

# Configure CORS
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Initialize database on startup
@app.on_event("startup")
async def startup_db_client():
    try:
        logger.info("Initializing database...")
        init_db()
        logger.info("Database initialized successfully")
        
        # Create storage directory if using local storage
        if settings.STORAGE_TYPE == "local":
            os.makedirs(settings.LOCAL_STORAGE_PATH, exist_ok=True)
            logger.info(f"Created storage directory: {settings.LOCAL_STORAGE_PATH}")
        
        # Validate OpenAI API key
        if not settings.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY is not set. Transcription service will not function properly.")
    
    except Exception as e:
        logger.error(f"Failed to initialize service: {str(e)}")
        # In a production environment, you might want to exit the application
        # if initialization fails
        raise

# Include API routers
app.include_router(
    transcriptions_router,
    prefix=f"{settings.API_V1_STR}/transcriptions",
    tags=["transcriptions"],
)

# Health check endpoints
@app.get("/health/ready")
async def health_ready():
    """Readiness probe for Kubernetes"""
    return {"status": "ready"}

@app.get("/health/live")
async def health_live():
    """Liveness probe for Kubernetes"""
    return {"status": "alive"}

# Internal metrics endpoint for service-to-service communication
@app.get("/internal/metrics", dependencies=[Depends(verify_service_api_key)])
async def get_metrics():
    """
    Return service metrics for internal monitoring
    This endpoint is protected by service API key authentication
    """
    return {
        "status": "ok",
        "uptime": time.time(),  # For demonstration; in production use actual uptime
        "version": "1.0.0",
    }

# Test OpenAI API key endpoint
@app.get("/internal/test-openai", dependencies=[Depends(verify_service_api_key)])
async def test_openai():
    """
    Test if the OpenAI API key is valid
    This endpoint is protected by service API key authentication
    """
    import openai
    try:
        # Set the API key
        openai.api_key = settings.OPENAI_API_KEY
        
        # Make a simple request to verify the key
        response = openai.Model.list()
        
        return {
            "status": "ok",
            "message": "OpenAI API key is valid",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"OpenAI API key validation failed: {str(e)}",
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    ) 