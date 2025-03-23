"""
Clip Generation Service API

This module provides the API endpoints for the Clip Generation Service.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Union, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, File, UploadFile, Form, Body, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator

from app.clip_generation.config.settings import settings
from app.clip_generation.services.clip_service import ClipGenerationService
from app.clip_generation.services.clip_generator import ClipGenerator
from app.clip_generation.workers.task_processor import TaskProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Clip Generation Service API",
    description="API for generating video clips from source videos.",
    version=settings.get("service_version", "1.0.0")
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get("allowed_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create service instances
clip_service = ClipGenerationService(
    output_dir=settings.get("output_dir"),
    temp_dir=settings.get("temp_dir"),
    max_concurrent_tasks=settings.get("max_concurrent_tasks"),
    default_format=settings.get("default_format"),
    default_quality=settings.get("default_quality")
)

# Create and start task processor
task_processor = TaskProcessor(
    output_dir=settings.get("output_dir"),
    temp_dir=settings.get("temp_dir"),
    num_workers=settings.get("worker_count"),
    ffmpeg_path=settings.get("ffmpeg_path")
)

# Start the task processor
@app.on_event("startup")
async def startup():
    task_processor.start()

# Stop the task processor
@app.on_event("shutdown")
async def shutdown():
    task_processor.stop()


# API Models
class ClipRequest(BaseModel):
    """Request model for creating a clip."""
    source_video: str = Field(..., description="Path to the source video file")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    output_name: Optional[str] = Field(None, description="Name for the output clip (without extension)")
    format: Optional[str] = Field(None, description="Output format (e.g., mp4, avi, mov)")
    quality: Optional[str] = Field(None, description="Quality setting (low, medium, high, ultra)")
    params: Optional[Dict[str, Any]] = Field(None, description="Additional parameters for clip generation")
    
    @validator("end_time")
    def end_time_must_be_greater_than_start_time(cls, v, values):
        if "start_time" in values and v <= values["start_time"]:
            raise ValueError("end_time must be greater than start_time")
        return v


class ClipResponse(BaseModel):
    """Response model for clip creation."""
    task_id: str = Field(..., description="Task ID")
    source_video: str = Field(..., description="Path to the source video file")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    output_name: str = Field(..., description="Name for the output clip")
    output_path: str = Field(..., description="Path to the output clip")
    format: str = Field(..., description="Output format")
    quality: str = Field(..., description="Quality setting")
    status: str = Field(..., description="Task status")
    created_at: float = Field(..., description="Task creation timestamp")


class TaskStatusResponse(BaseModel):
    """Response model for task status."""
    task_id: str = Field(..., description="Task ID")
    status: str = Field(..., description="Task status")
    progress: Optional[float] = Field(None, description="Task progress (0-100)")
    created_at: float = Field(..., description="Task creation timestamp")
    updated_at: float = Field(..., description="Task last update timestamp")
    completed_at: Optional[float] = Field(None, description="Task completion timestamp")
    error: Optional[str] = Field(None, description="Error message if task failed")


class TaskListResponse(BaseModel):
    """Response model for task list."""
    tasks: List[TaskStatusResponse] = Field(..., description="List of tasks")
    total: int = Field(..., description="Total number of tasks")
    limit: int = Field(..., description="Maximum number of tasks returned")
    offset: int = Field(..., description="Offset for pagination")


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.get("service_name"),
        "version": settings.get("service_version"),
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "time": time.time(),
        "queue_stats": task_processor.get_queue_stats()
    }


@app.post("/clips", response_model=ClipResponse)
async def create_clip(clip_request: ClipRequest, background_tasks: BackgroundTasks):
    """
    Create a new clip from the source video.
    
    Args:
        clip_request: Clip generation request
        background_tasks: Background tasks manager
    
    Returns:
        Clip creation response
    """
    try:
        # Create the clip task
        task = clip_service.create_clip_task(
            source_video=clip_request.source_video,
            start_time=clip_request.start_time,
            end_time=clip_request.end_time,
            output_name=clip_request.output_name,
            format=clip_request.format,
            quality=clip_request.quality,
            params=clip_request.params
        )
        
        # Add task to the processor queue
        background_tasks.add_task(task_processor.add_task, task)
        
        return task
        
    except Exception as e:
        logger.error(f"Error creating clip: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating clip: {str(e)}")


@app.get("/clips/{task_id}", response_model=TaskStatusResponse)
async def get_clip_status(task_id: str):
    """
    Get the status of a clip generation task.
    
    Args:
        task_id: Task ID
    
    Returns:
        Task status response
    """
    try:
        task_status = clip_service.get_task_status(task_id)
        
        # Convert to response model
        return TaskStatusResponse(
            task_id=task_status["task_id"],
            status=task_status["status"],
            progress=task_status.get("progress"),
            created_at=task_status["created_at"],
            updated_at=task_status["updated_at"],
            completed_at=task_status.get("completed_at"),
            error=task_status.get("error")
        )
        
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    except Exception as e:
        logger.error(f"Error getting task status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting task status: {str(e)}")


@app.get("/clips", response_model=TaskListResponse)
async def list_clips(
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    List clip generation tasks.
    
    Args:
        status: Filter by task status
        limit: Maximum number of tasks to return
        offset: Offset for pagination
    
    Returns:
        Task list response
    """
    try:
        tasks = clip_service.list_tasks(status=status, limit=limit, offset=offset)
        
        # Convert to response model
        task_responses = []
        for task in tasks:
            task_responses.append(TaskStatusResponse(
                task_id=task["task_id"],
                status=task["status"],
                progress=task.get("progress"),
                created_at=task["created_at"],
                updated_at=task["updated_at"],
                completed_at=task.get("completed_at"),
                error=task.get("error")
            ))
        
        return TaskListResponse(
            tasks=task_responses,
            total=len(tasks),
            limit=limit,
            offset=offset
        )
        
    except Exception as e:
        logger.error(f"Error listing tasks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing tasks: {str(e)}")


@app.delete("/clips/{task_id}")
async def cancel_clip(task_id: str):
    """
    Cancel a clip generation task.
    
    Args:
        task_id: Task ID
    
    Returns:
        Task cancellation response
    """
    try:
        task = clip_service.cancel_task(task_id)
        
        return {
            "task_id": task_id,
            "status": "cancelled",
            "message": f"Task {task_id} cancelled successfully"
        }
        
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Error cancelling task: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error cancelling task: {str(e)}")


@app.get("/clips/{task_id}/download")
async def download_clip(task_id: str):
    """
    Download a completed clip.
    
    Args:
        task_id: Task ID
    
    Returns:
        File response with the clip
    """
    try:
        task = clip_service.get_task_status(task_id)
        
        # Check if task is completed
        if task["status"] != "completed":
            raise HTTPException(
                status_code=400, 
                detail=f"Clip not ready for download, current status: {task['status']}"
            )
        
        # Check if output file exists
        output_path = task["output_path"]
        if not os.path.exists(output_path):
            raise HTTPException(
                status_code=404, 
                detail=f"Clip file not found: {output_path}"
            )
        
        # Return file response
        return FileResponse(
            path=output_path,
            filename=os.path.basename(output_path),
            media_type=f"video/{task['format']}"
        )
        
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error downloading clip: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error downloading clip: {str(e)}")


# Advanced API endpoints

@app.post("/clips/extract-frame")
async def extract_frame(
    source_video: str = Form(..., description="Path to the source video file"),
    timestamp: float = Form(..., description="Time in seconds for the frame to extract"),
    format: str = Form("jpg", description="Output image format (jpg, png, etc.)"),
    quality: int = Form(90, description="Image quality (0-100) for lossy formats"),
    width: Optional[int] = Form(None, description="Width for resizing"),
    height: Optional[int] = Form(None, description="Height for resizing")
):
    """
    Extract a single frame from a video at the specified timestamp.
    
    Args:
        source_video: Path to the source video file
        timestamp: Time in seconds for the frame to extract
        format: Output image format (jpg, png, etc.)
        quality: Image quality (0-100) for lossy formats
        width: Width for resizing
        height: Height for resizing
    
    Returns:
        Path to the extracted frame
    """
    try:
        # Create clip generator
        clip_generator = ClipGenerator(
            ffmpeg_path=settings.get("ffmpeg_path"),
            temp_dir=settings.get("temp_dir")
        )
        
        # Determine resize dimensions
        resize = None
        if width is not None and height is not None:
            resize = (width, height)
        
        # Generate output path
        base_name = os.path.splitext(os.path.basename(source_video))[0]
        output_name = f"{base_name}_frame_{timestamp:.1f}.{format}"
        output_path = os.path.join(settings.get("output_dir"), output_name)
        
        # Extract frame
        result_path = clip_generator.extract_frame(
            source_video=source_video,
            output_path=output_path,
            timestamp=timestamp,
            format=format,
            quality=quality,
            resize=resize
        )
        
        return {
            "source_video": source_video,
            "timestamp": timestamp,
            "output_path": result_path,
            "format": format,
            "quality": quality,
            "resize": resize
        }
        
    except Exception as e:
        logger.error(f"Error extracting frame: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error extracting frame: {str(e)}")


@app.get("/clips/video-info")
async def get_video_info(video_path: str = Query(..., description="Path to the video file")):
    """
    Get information about a video file.
    
    Args:
        video_path: Path to the video file
    
    Returns:
        Dictionary containing video information
    """
    try:
        # Create clip generator
        clip_generator = ClipGenerator(
            ffmpeg_path=settings.get("ffmpeg_path"),
            temp_dir=settings.get("temp_dir")
        )
        
        # Get video info
        video_info = clip_generator.get_video_info(video_path)
        
        return video_info
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Video file not found: {video_path}")
    
    except Exception as e:
        logger.error(f"Error getting video info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting video info: {str(e)}")


@app.get("/settings")
async def get_settings():
    """
    Get service settings.
    
    Returns:
        Dictionary of service settings (with sensitive information removed)
    """
    # Get settings and remove sensitive information
    service_settings = settings.to_dict()
    
    # Remove sensitive settings
    sensitive_keys = ["api_key", "access_token", "secret_key", "password"]
    for key in sensitive_keys:
        if key in service_settings:
            service_settings[key] = "***REDACTED***"
    
    return service_settings 