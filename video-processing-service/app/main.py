"""
Main FastAPI application for the Video Processing Service.
"""

import os
import logging
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.staticfiles import StaticFiles
import time

from app.config import settings
from app.api.endpoints import subtitles, batch, audio, music

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create the FastAPI application
app = FastAPI(
    title="Video Processing Service",
    description="API for video processing, subtitle generation, and audio enhancement",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Exception handling middleware
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception in {request.url.path}: {str(exc)}", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )

# Include API routers
app.include_router(subtitles.router, prefix=settings.API_V1_PREFIX)
app.include_router(batch.router, prefix=settings.API_V1_PREFIX)
app.include_router(audio.router, prefix=settings.API_V1_PREFIX)
app.include_router(music.router, prefix=settings.API_V1_PREFIX)

# Create temporary directory
os.makedirs(settings.TEMP_DIRECTORY, exist_ok=True)

# Mount static directory for output files
static_path = os.path.join(settings.TEMP_DIRECTORY, "outputs")
os.makedirs(static_path, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/", tags=["status"])
async def root():
    """
    Root endpoint returning API status.
    """
    return {
        "status": "running",
        "service": "Video Processing Service",
        "version": "1.0.0",
    }

@app.get("/health", tags=["status"])
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}

@app.get("/info", tags=["status"])
async def service_info():
    """
    Service information endpoint.
    """
    return {
        "service": "Video Processing Service",
        "version": "1.0.0",
        "endpoints": [
            f"{settings.API_V1_PREFIX}/subtitles",
            f"{settings.API_V1_PREFIX}/batch",
            f"{settings.API_V1_PREFIX}/audio",
            f"{settings.API_V1_PREFIX}/music"
        ],
        "features": [
            "Subtitle generation in multiple formats (SRT, VTT, ASS)",
            "Burnt-in subtitles in video",
            "Optimized subtitle positioning",
            "Multi-language support",
            "Batch processing",
            "Smart text breaking for readability",
            "Duration calibration for reading speed",
            "Emphasis detection",
            "Audio enhancement and noise reduction",
            "Voice clarity improvement",
            "Environmental sound classification"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 