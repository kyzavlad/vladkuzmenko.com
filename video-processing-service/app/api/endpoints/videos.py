from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks, status, Query
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import os
import uuid
import logging
from datetime import datetime

from app.db.postgres import get_db
from app.core.config import settings
from app.models.video import Video, ProcessingStatus
from app.schemas.video import (
    VideoCreate, VideoResponse, VideoList, VideoDetail,
    ProcessingOptions
)
from app.services.video_service import VideoService
from app.services.storage_service import StorageService
from app.services.auth import get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload", response_model=VideoResponse, status_code=status.HTTP_201_CREATED)
async def upload_video(
    background_tasks: BackgroundTasks,
    title: str = Form(...),
    description: Optional[str] = Form(None),
    file: UploadFile = File(...),
    transcription: bool = Form(True),
    subtitles: bool = Form(True),
    enhancement: bool = Form(False),
    avatar: bool = Form(False),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Upload a video file and optionally start processing tasks.
    """
    user_id = current_user["user_id"]
    
    # Check file extension
    file_ext = os.path.splitext(file.filename)[1].lower().lstrip(".")
    if file_ext not in settings.ALLOWED_VIDEO_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file format. Allowed formats: {', '.join(settings.ALLOWED_VIDEO_FORMATS)}"
        )
    
    # Check file size
    content_length = int(file.size)
    if content_length > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE_MB}MB"
        )
    
    # Create video service
    video_service = VideoService(db)
    storage_service = StorageService()
    
    # Generate unique filename and storage path
    file_id = str(uuid.uuid4())
    storage_filename = f"{file_id}.{file_ext}"
    
    # Create video record
    video_data = VideoCreate(
        user_id=user_id,
        title=title,
        description=description,
        original_filename=file.filename,
        content_type=file.content_type,
        file_size=content_length,
        transcription_enabled=transcription,
        subtitles_enabled=subtitles,
        enhancement_enabled=enhancement,
        avatar_enabled=avatar
    )
    
    # Save video record to database
    try:
        video = await video_service.create_video(
            video_data, 
            storage_path=f"videos/{user_id}/{storage_filename}"
        )
        
        # Queue background task to save file to storage and process video
        background_tasks.add_task(
            video_service.process_uploaded_video,
            video.id,
            file,
            ProcessingOptions(
                transcription=transcription,
                subtitles=subtitles,
                enhancement=enhancement,
                avatar=avatar
            )
        )
        
        return VideoResponse(
            id=video.id,
            title=video.title,
            description=video.description,
            status=video.status,
            created_at=video.created_at,
            message="Video uploaded successfully. Processing started in the background."
        )
    except Exception as e:
        logger.error(f"Error uploading video: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload video. Please try again."
        )


@router.get("/", response_model=VideoList)
async def list_videos(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    status: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    List videos for the current user.
    """
    user_id = current_user["user_id"]
    video_service = VideoService(db)
    
    videos, total = await video_service.get_videos_by_user(
        user_id, skip=skip, limit=limit, status=status
    )
    
    return VideoList(
        videos=[
            VideoResponse(
                id=video.id,
                title=video.title,
                description=video.description,
                status=video.status,
                created_at=video.created_at,
                progress=video.progress
            ) for video in videos
        ],
        total=total
    )


@router.get("/{video_id}", response_model=VideoDetail)
async def get_video(
    video_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get detailed information about a specific video.
    """
    user_id = current_user["user_id"]
    video_service = VideoService(db)
    
    video = await video_service.get_video(video_id)
    
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found"
        )
    
    if video.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this video"
        )
    
    # Get presigned URLs for video files
    storage_service = StorageService()
    
    original_url = await storage_service.get_presigned_url(video.storage_path)
    enhanced_url = None
    avatar_url = None
    subtitle_url = None
    
    if video.enhanced_video_path:
        enhanced_url = await storage_service.get_presigned_url(video.enhanced_video_path)
    
    if video.avatar_video_path:
        avatar_url = await storage_service.get_presigned_url(video.avatar_video_path)
    
    if video.subtitle_path:
        subtitle_url = await storage_service.get_presigned_url(video.subtitle_path)
    
    return VideoDetail(
        id=video.id,
        title=video.title,
        description=video.description,
        original_filename=video.original_filename,
        content_type=video.content_type,
        file_size=video.file_size,
        duration=video.duration,
        status=video.status,
        progress=video.progress,
        created_at=video.created_at,
        updated_at=video.updated_at,
        processed_at=video.processed_at,
        transcription_enabled=video.transcription_enabled,
        subtitles_enabled=video.subtitles_enabled,
        enhancement_enabled=video.enhancement_enabled,
        avatar_enabled=video.avatar_enabled,
        original_url=original_url,
        enhanced_url=enhanced_url,
        avatar_url=avatar_url,
        subtitle_url=subtitle_url,
        error_message=video.error_message,
        transcription_id=video.transcription_id,
        tasks=[
            {
                "id": task.id,
                "task_type": task.task_type,
                "status": task.status,
                "progress": task.progress,
                "created_at": task.created_at,
                "started_at": task.started_at,
                "completed_at": task.completed_at,
                "error_message": task.error_message
            } for task in video.tasks
        ]
    )


@router.delete("/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_video(
    video_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Delete a video and all associated resources.
    """
    user_id = current_user["user_id"]
    video_service = VideoService(db)
    
    video = await video_service.get_video(video_id)
    
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found"
        )
    
    if video.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this video"
        )
    
    # Delete all associated files and resources
    try:
        await video_service.delete_video(video_id)
        return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content={})
    except Exception as e:
        logger.error(f"Error deleting video {video_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete video. Please try again."
        ) 