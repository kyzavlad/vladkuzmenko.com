from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
import shutil
import os
from typing import Optional, List
import logging
from pydantic import BaseModel
from clip_generator import ClipGenerator
from .models.config import AppConfig
from .models.job import JobCreate, JobResponse, JobStatus
from .services.manager import ServiceManager
from .database.dependencies import get_db_session
from .auth.dependencies import get_current_user
from .models.user import User
import stripe

from .api import jobs, users, token, stripe_webhook, monitoring, quality
from .config import settings
from .database.session import engine
from .database.models import Base

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize Stripe
stripe.api_key = settings.STRIPE_SECRET_KEY

# Create FastAPI app
app = FastAPI(
    title="Clip Generation Service",
    description="A service for generating video clips with advanced features",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service manager
service_manager: Optional[ServiceManager] = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global service_manager
    try:
        # Load configuration
        config = AppConfig()  # You'll need to implement this
        
        # Initialize service manager
        service_manager = ServiceManager(app, config)
        await service_manager.start()
        
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Error starting application: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up services on shutdown."""
    global service_manager
    if service_manager:
        await service_manager.stop()
        logger.info("Application stopped successfully")

class ClipRequest(BaseModel):
    target_duration: Optional[float] = 30.0
    target_width: Optional[int] = 1080
    target_height: Optional[int] = 1920
    target_lufs: Optional[float] = -14.0

@app.post("/generate-clip/")
async def generate_clip(
    file: UploadFile = File(...),
    request: ClipRequest = ClipRequest()
):
    """
    Generate an optimized clip from the uploaded video.
    """
    try:
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the clip
        generator = ClipGenerator(file_path)
        output_path, duration = generator.process_clip(
            target_duration=request.target_duration
        )
        
        # Return the processed file
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=f"processed_{file.filename}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/jobs", response_model=JobResponse)
async def create_job(
    job_data: JobCreate,
    current_user: User = Depends(get_current_user)
):
    """Create a new job."""
    try:
        job_data.user_id = current_user.id
        job = await service_manager.submit_job(job_data.dict())
        return JobResponse(**job)
    except Exception as e:
        logger.error(f"Error creating job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get job status."""
    try:
        job = await service_manager.get_job_status(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Check if user owns the job
        if job["user_id"] != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to access this job")
        
        return JobResponse(**job)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs", response_model=List[JobResponse])
async def list_jobs(
    skip: int = 0,
    limit: int = 100,
    status: Optional[JobStatus] = None,
    current_user: User = Depends(get_current_user)
):
    """List user's jobs."""
    try:
        # TODO: Implement job listing
        return []
    except Exception as e:
        logger.error(f"Error listing jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/jobs/{job_id}")
async def cancel_job(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """Cancel a job."""
    try:
        # Check if job exists and user owns it
        job = await service_manager.get_job_status(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job["user_id"] != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to cancel this job")
        
        success = await service_manager.cancel_job(job_id)
        if not success:
            raise HTTPException(status_code=400, detail="Could not cancel job")
        
        return {"message": "Job cancelled successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str,
    current_user: User = Depends(get_current_user)
):
    """WebSocket endpoint for real-time job updates."""
    # Verify user owns the WebSocket connection
    if str(current_user.id) != user_id:
        await websocket.close(code=4003, reason="Not authorized")
        return
    
    await service_manager.websocket_manager.handle_client(websocket, user_id)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Clip Generation Service API",
        "version": "1.0.0",
        "status": "operational"
    }

# Include routers
app.include_router(jobs.router)
app.include_router(users.router)
app.include_router(token.router)
app.include_router(stripe_webhook.router)
app.include_router(monitoring.router)
app.include_router(quality.router)

# Error handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    ) 