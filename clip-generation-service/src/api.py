from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import os
from typing import List, Optional
import shutil
from datetime import datetime

from .models import (
    ClipGenerationRequest,
    ClipUpdateRequest,
    BatchExportRequest,
    JobStatus,
    ClipStatus,
    Clip
)
from .job_manager import JobManager

app = FastAPI(title="Clip Generation Service")
job_manager = JobManager()

# Create necessary directories
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
PREVIEW_DIR = "previews"
THUMBNAIL_DIR = "thumbnails"

for directory in [UPLOAD_DIR, OUTPUT_DIR, PREVIEW_DIR, THUMBNAIL_DIR]:
    os.makedirs(directory, exist_ok=True)

def save_upload_file(upload_file: UploadFile) -> str:
    """Save uploaded file and return the path."""
    file_path = os.path.join(UPLOAD_DIR, upload_file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return file_path

def estimate_completion_time(file_size: int, request: ClipGenerationRequest) -> int:
    """Estimate job completion time in seconds."""
    # Simple estimation based on file size and number of clips
    base_time = file_size / (1024 * 1024)  # 1 second per MB
    return int(base_time * request.max_clips)

@app.post("/api/video/generate-clips")
async def generate_clips(
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(...),
    request: ClipGenerationRequest = ClipGenerationRequest()
) -> JSONResponse:
    """
    Generate clips from uploaded video.
    """
    try:
        # Save uploaded file
        video_path = save_upload_file(video_file)
        
        # Create job
        job_id = job_manager.create_job(video_path, request.dict())
        
        # Start processing in background
        background_tasks.add_task(job_manager.process_job, job_id)
        
        # Estimate completion time
        file_size = os.path.getsize(video_path)
        estimated_time = estimate_completion_time(file_size, request)
        
        return JSONResponse({
            "job_id": job_id,
            "estimated_completion_time": estimated_time
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/video/generate-clips/status/{job_id}")
async def get_job_status(job_id: str) -> JobStatus:
    """
    Get status of a clip generation job.
    """
    status = job_manager.get_job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return status

@app.put("/api/video/clips/{clip_id}")
async def update_clip(
    clip_id: str,
    request: ClipUpdateRequest
) -> JSONResponse:
    """
    Update clip metadata.
    """
    # Find the clip in all jobs
    for job in job_manager.jobs.values():
        for clip in job["status"].clips:
            if clip.clip_id == clip_id:
                updated_fields = []
                
                # Update fields if provided
                if request.start_time is not None:
                    clip.start_time = request.start_time
                    updated_fields.append("start_time")
                    
                if request.end_time is not None:
                    clip.end_time = request.end_time
                    updated_fields.append("end_time")
                    
                if request.title is not None:
                    clip.title = request.title
                    updated_fields.append("title")
                    
                if request.description is not None:
                    clip.description = request.description
                    updated_fields.append("description")
                    
                if request.tags is not None:
                    clip.tags = request.tags
                    updated_fields.append("tags")
                    
                return JSONResponse({
                    "clip_id": clip_id,
                    "status": "updated",
                    "updated_fields": updated_fields
                })
                
    raise HTTPException(status_code=404, detail="Clip not found")

@app.delete("/api/video/clips/{clip_id}")
async def delete_clip(clip_id: str) -> JSONResponse:
    """
    Delete a clip.
    """
    # Find and delete the clip in all jobs
    for job in job_manager.jobs.values():
        for i, clip in enumerate(job["status"].clips):
            if clip.clip_id == clip_id:
                # Remove clip from list
                job["status"].clips.pop(i)
                job["status"].clips_generated -= 1
                
                # Delete associated files
                for url_attr in ["preview_url", "download_url", "thumbnail_url"]:
                    file_path = getattr(clip, url_attr).lstrip("/")
                    if os.path.exists(file_path):
                        os.remove(file_path)
                
                return JSONResponse({
                    "success": True,
                    "message": f"Clip {clip_id} deleted successfully"
                })
                
    raise HTTPException(status_code=404, detail="Clip not found")

@app.post("/api/video/clips/batch-export")
async def batch_export(
    background_tasks: BackgroundTasks,
    request: BatchExportRequest
) -> JSONResponse:
    """
    Export multiple clips with platform-specific settings.
    """
    # Validate clip IDs
    clips_to_export = []
    for clip_id in request.clip_ids:
        clip_found = False
        for job in job_manager.jobs.values():
            for clip in job["status"].clips:
                if clip.clip_id == clip_id:
                    clips_to_export.append(clip)
                    clip_found = True
                    break
            if clip_found:
                break
        if not clip_found:
            raise HTTPException(
                status_code=404,
                detail=f"Clip {clip_id} not found"
            )
    
    # Create export job
    job_id = job_manager.create_job(
        video_path="",  # Not needed for export
        request_data={
            "clips": [clip.dict() for clip in clips_to_export],
            "platform": request.platform,
            "include_captions": request.include_captions,
            "caption_style": request.caption_style,
            "add_watermark": request.add_watermark,
            "watermark_image": request.watermark_image
        }
    )
    
    # TODO: Implement actual export processing in background
    
    return JSONResponse({
        "batch_job_id": job_id,
        "estimated_completion_time": len(clips_to_export) * 30  # 30 seconds per clip
    }) 