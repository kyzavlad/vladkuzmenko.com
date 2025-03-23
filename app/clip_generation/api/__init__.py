"""
Clip Generation Service API

This module defines the main API router for the Clip Generation Service,
including endpoints for clip generation, face tracking, and other features.
"""

from fastapi import APIRouter
from app.clip_generation.api.face_tracking_api import router as face_tracking_router

# Main API router
api_router = APIRouter()

# Include all service routers
api_router.include_router(face_tracking_router)

# Define root API endpoint
@api_router.get("/")
async def root():
    """Root endpoint for the Clip Generation Service API."""
    return {
        "service": "Clip Generation Service",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs",
        "features": [
            "Clip Generation",
            "Face Tracking",
            "Smart Framing"
        ]
    } 