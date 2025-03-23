#!/usr/bin/env python3
"""
Clip Generation Service

This is the main entry point for the Clip Generation Service, which provides
video clip generation capabilities with advanced features such as face tracking,
smart framing, and more.

To run the service:
    uvicorn app.clip_generation.main:app --reload

For production:
    gunicorn -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:8000 app.clip_generation.main:app
"""

import os
import logging
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi

from app.clip_generation.api import api_router
from app.clip_generation.config.settings import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("clip_generation_service")

# Load settings
settings = Settings()

# Create FastAPI app
app = FastAPI(
    title="Clip Generation Service",
    description="API for video clip generation with face tracking and smart framing",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url="/api/v1/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI route."""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )


@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    """Custom ReDoc route."""
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - ReDoc",
        redoc_js_url="/static/redoc.standalone.js",
    )


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware to add process time header to responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    logger.info("Starting Clip Generation Service...")
    
    # Create necessary directories
    os.makedirs(settings.output_dir, exist_ok=True)
    os.makedirs(settings.temp_dir, exist_ok=True)
    
    logger.info(f"Output directory: {settings.output_dir}")
    logger.info(f"Temp directory: {settings.temp_dir}")
    logger.info(f"Models path: {settings.models_path}")
    
    logger.info("Clip Generation Service started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    logger.info("Shutting down Clip Generation Service...")
    logger.info("Cleanup completed")


# Include API router
app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Clip Generation Service",
        "version": "1.0.0",
        "status": "operational",
        "api_docs": "/docs",
        "api_root": "/api/v1"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "clip_generation"
    }


# Serve static files
try:
    app.mount("/static", StaticFiles(directory="app/clip_generation/static"), name="static")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")
    # Create minimal static directory
    os.makedirs("app/clip_generation/static", exist_ok=True)
    app.mount("/static", StaticFiles(directory="app/clip_generation/static"), name="static")


# Custom exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    import traceback
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "message": "Internal server error",
            "detail": str(exc) if settings.debug else "An unexpected error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 