from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
import os
import shutil
import uuid
import datetime
import subprocess
import json

from app.core.config import settings
from app.db.session import get_db
from app.models.video import Video, ProcessingTask
from app.schemas.video import VideoCreate, VideoResponse, VideoProcessingOptions
from app.services.storage import save_video, get_video_info
from app.services.auth import get_current_user_id, verify_service_api_key
from app.services.task_queue import queue_processing_task

router = APIRouter()

@router.post("/upload", response_model=VideoResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    title: str = Form(...),
    description: Optional[str] = Form(None),
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """
    Upload a new video file and store it for processing
    """
    # Validate file extension
    file_ext = file.filename.split('.')[-1].lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Allowed formats: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )
    
    # Generate a unique filename
    unique_filename = f"{uuid.uuid4()}.{file_ext}"
    storage_path = f"{settings.LOCAL_STORAGE_PATH}/{unique_filename}"
    
    # Ensure storage directory exists
    os.makedirs(os.path.dirname(storage_path), exist_ok=True)
    
    # Save uploaded file
    try:
        with open(storage_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store video: {str(e)}")
    
    # Get video information using FFmpeg
    try:
        video_info = get_video_info(storage_path)
    except Exception as e:
        # Clean up the file if extraction fails
        os.remove(storage_path)
        raise HTTPException(status_code=400, detail=f"Invalid video file: {str(e)}")
    
    # Create video record in database
    new_video = Video(
        user_id=user_id,
        title=title,
        description=description,
        original_filename=file.filename,
        storage_path=storage_path,
        file_size=os.path.getsize(storage_path),
        duration=video_info.get("duration"),
        format=file_ext,
        width=video_info.get("width"),
        height=video_info.get("height"),
        fps=video_info.get("fps"),
        status="uploaded"
    )
    
    db.add(new_video)
    db.commit()
    db.refresh(new_video)
    
    # Return the created video information
    return VideoResponse(
        id=new_video.id,
        title=new_video.title,
        description=new_video.description,
        original_filename=new_video.original_filename,
        file_size=new_video.file_size,
        duration=new_video.duration,
        format=new_video.format,
        width=new_video.width,
        height=new_video.height,
        fps=new_video.fps,
        status=new_video.status,
        created_at=new_video.created_at
    )

@router.post("/{video_id}/process", response_model=Dict[str, Any])
async def process_video(
    video_id: str,
    processing_options: VideoProcessingOptions,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """
    Start processing a previously uploaded video with specified options
    """
    # Find the video in the database
    video = db.query(Video).filter(Video.id == video_id, Video.user_id == user_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Validate video status
    if video.status not in ["uploaded", "failed"]:
        raise HTTPException(status_code=400, detail=f"Video cannot be processed in '{video.status}' status")
    
    # Update video status
    video.status = "processing"
    video.processing_options = processing_options.dict()
    video.processing_start_time = datetime.datetime.utcnow()
    db.commit()
    
    # Queue processing tasks based on selected options
    tasks = []
    
    if processing_options.transcription:
        task = ProcessingTask(
            video_id=video.id,
            task_type="transcription",
            parameters={"language": processing_options.transcription_language}
        )
        db.add(task)
        tasks.append(task)
    
    if processing_options.pause_detection:
        task = ProcessingTask(
            video_id=video.id,
            task_type="pause_detection",
            parameters={"min_pause_duration": processing_options.min_pause_duration}
        )
        db.add(task)
        tasks.append(task)
    
    if processing_options.subtitles:
        task = ProcessingTask(
            video_id=video.id,
            task_type="subtitles",
            parameters={
                "font": processing_options.subtitle_font,
                "font_size": processing_options.subtitle_font_size,
                "font_color": processing_options.subtitle_font_color,
                "format": processing_options.subtitle_format
            }
        )
        db.add(task)
        tasks.append(task)
    
    # Add more tasks for other processing options
    
    db.commit()
    
    # Queue tasks for processing
    for task in tasks:
        background_tasks.add_task(queue_processing_task, task.id)
    
    return {
        "id": video.id,
        "status": "processing",
        "tasks_queued": len(tasks),
        "processing_options": processing_options.dict()
    }

@router.get("/{video_id}", response_model=VideoResponse)
async def get_video(
    video_id: str,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """
    Get information about a specific video
    """
    video = db.query(Video).filter(Video.id == video_id, Video.user_id == user_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return VideoResponse.from_orm(video)

@router.get("/", response_model=List[VideoResponse])
async def list_videos(
    skip: int = 0,
    limit: int = 100,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """
    List all videos for the current user
    """
    videos = db.query(Video).filter(Video.user_id == user_id).offset(skip).limit(limit).all()
    return [VideoResponse.from_orm(video) for video in videos] 