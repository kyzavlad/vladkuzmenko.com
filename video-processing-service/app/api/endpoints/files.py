import logging
import os
import uuid
import asyncio
from typing import List
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File, Query, Path
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgres import get_db
from app.models.video import ProcessingStatus
from app.services.video_service import VideoService
from app.services.storage_service import StorageService
from app.services.transcription_service import TranscriptionService
from app.schemas.video import VideoCreate, VideoResponse, VideoDetail, ProcessingOptions
from app.core.config import settings
from app.api.deps import get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload", response_model=VideoResponse, status_code=202)
async def upload_video(
    background_tasks: BackgroundTasks,
    title: str = Query(..., description="Video title"),
    description: str = Query(None, description="Video description"),
    transcription: bool = Query(True, description="Enable transcription"),
    subtitles: bool = Query(True, description="Enable subtitles"),
    enhancement: bool = Query(False, description="Enable video enhancement"),
    avatar: bool = Query(False, description="Enable avatar generation"),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Upload a new video file for processing.
    
    This endpoint initiates upload and async processing of a video.
    The processing happens in the background after the API returns.
    """
    # Validate file
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Generate storage path
    user_id = current_user["sub"]
    file_ext = os.path.splitext(file.filename)[1]
    if not file_ext:
        # Default to mp4 if no extension
        file_ext = ".mp4"
    
    storage_path = f"videos/{user_id}/{uuid.uuid4()}{file_ext}"
    
    # Initialize services
    video_service = VideoService(db)
    
    # Create video record
    video_data = VideoCreate(
        user_id=user_id,
        title=title,
        description=description,
        original_filename=file.filename,
        content_type=file.content_type,
        file_size=0,  # Will be updated after upload
        transcription_enabled=transcription,
        subtitles_enabled=subtitles,
        enhancement_enabled=enhancement,
        avatar_enabled=avatar
    )
    
    # Create video record
    video = await video_service.create_video(video_data, storage_path)
    
    # Schedule background processing
    processing_options = ProcessingOptions(
        transcription=transcription,
        subtitles=subtitles,
        enhancement=enhancement,
        avatar=avatar
    )
    
    # Add processing task to background
    background_tasks.add_task(
        video_service.process_uploaded_video,
        video.id,
        file,
        processing_options
    )
    
    # If transcription is enabled, schedule it
    if transcription:
        transcription_service = TranscriptionService(db)
        background_tasks.add_task(
            transcription_service.request_transcription,
            video.id
        )
    
    # Return response with video ID
    return VideoResponse(
        id=video.id,
        title=video.title,
        description=video.description,
        status=video.status,
        created_at=video.created_at,
        progress=0.0,
        message="Video upload started"
    )


@router.get("/{video_id}/download", response_class=StreamingResponse)
async def download_video(
    video_id: str = Path(..., description="Video ID"),
    type: str = Query("original", description="Video type (original, enhanced, avatar)"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Download a processed video file.
    
    Args:
        video_id: ID of the video to download
        type: Type of video to download (original, enhanced, avatar)
    """
    # Initialize services
    video_service = VideoService(db)
    storage_service = StorageService()
    
    # Get video
    video = await video_service.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Check if user owns the video
    if video.user_id != current_user["sub"]:
        raise HTTPException(status_code=403, detail="Not authorized to access this video")
    
    # Determine which file to return
    if type == "original":
        storage_path = video.storage_path
        filename = f"original_{os.path.basename(video.storage_path)}"
    elif type == "enhanced":
        if not video.enhanced_video_path:
            raise HTTPException(status_code=404, detail="Enhanced video not available")
        storage_path = video.enhanced_video_path
        filename = f"enhanced_{os.path.basename(video.storage_path)}"
    elif type == "avatar":
        if not video.avatar_video_path:
            raise HTTPException(status_code=404, detail="Avatar video not available")
        storage_path = video.avatar_video_path
        filename = f"avatar_{os.path.basename(video.storage_path)}"
    else:
        raise HTTPException(status_code=400, detail="Invalid video type")
    
    try:
        # Get file content
        file_content = await storage_service.get_file(storage_path)
        
        # For local storage, file_content is a path, so we need to read it
        if isinstance(file_content, str) and os.path.exists(file_content):
            async def file_iterator():
                async with open(file_content, mode="rb") as f:
                    while chunk := await f.read(8192):
                        yield chunk
            
            return StreamingResponse(
                file_iterator(),
                media_type=video.content_type,
                headers={"Content-Disposition": f'attachment; filename="{filename}"'}
            )
        
        # For S3 storage, file_content is bytes
        return StreamingResponse(
            iter([file_content]),
            media_type=video.content_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        logger.error(f"Error downloading video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error downloading video")


@router.get("/{video_id}/subtitle", response_class=StreamingResponse)
async def download_subtitles(
    video_id: str = Path(..., description="Video ID"),
    format: str = Query("srt", description="Subtitle format (srt, vtt)"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Download the subtitle file for a video.
    
    Args:
        video_id: ID of the video
        format: Subtitle format (srt, vtt)
    """
    # Initialize services
    video_service = VideoService(db)
    storage_service = StorageService()
    
    # Get video
    video = await video_service.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Check if user owns the video
    if video.user_id != current_user["sub"]:
        raise HTTPException(status_code=403, detail="Not authorized to access this video")
    
    # Check if subtitles are available
    if not video.subtitle_path:
        raise HTTPException(status_code=404, detail="Subtitles not available")
    
    # Get file path and content type based on format
    if format == "srt":
        file_path = video.subtitle_path.replace(".vtt", ".srt")
        content_type = "application/x-subrip"
        filename = f"{os.path.splitext(os.path.basename(video.storage_path))[0]}.srt"
    elif format == "vtt":
        file_path = video.subtitle_path
        content_type = "text/vtt"
        filename = f"{os.path.splitext(os.path.basename(video.storage_path))[0]}.vtt"
    else:
        raise HTTPException(status_code=400, detail="Invalid subtitle format")
    
    try:
        # Get file content
        file_content = await storage_service.get_file(file_path)
        
        # For local storage, file_content is a path
        if isinstance(file_content, str) and os.path.exists(file_content):
            async def file_iterator():
                async with open(file_content, mode="rb") as f:
                    while chunk := await f.read(8192):
                        yield chunk
            
            return StreamingResponse(
                file_iterator(),
                media_type=content_type,
                headers={"Content-Disposition": f'attachment; filename="{filename}"'}
            )
        
        # For S3 storage, file_content is bytes
        return StreamingResponse(
            iter([file_content]),
            media_type=content_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Subtitle file in {format} format not found")
    except Exception as e:
        logger.error(f"Error downloading subtitles for video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error downloading subtitles")


@router.get("/{video_id}/presigned-url", response_model=dict)
async def get_presigned_url(
    video_id: str = Path(..., description="Video ID"),
    type: str = Query("original", description="Video type (original, enhanced, avatar)"),
    expires: int = Query(3600, description="URL expiry time in seconds"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get a presigned URL for accessing a video file.
    
    Args:
        video_id: ID of the video
        type: Type of video (original, enhanced, avatar)
        expires: URL expiry time in seconds
    """
    # Initialize services
    video_service = VideoService(db)
    storage_service = StorageService()
    
    # Get video
    video = await video_service.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Check if user owns the video
    if video.user_id != current_user["sub"]:
        raise HTTPException(status_code=403, detail="Not authorized to access this video")
    
    # Determine which file to generate URL for
    if type == "original":
        storage_path = video.storage_path
    elif type == "enhanced":
        if not video.enhanced_video_path:
            raise HTTPException(status_code=404, detail="Enhanced video not available")
        storage_path = video.enhanced_video_path
    elif type == "avatar":
        if not video.avatar_video_path:
            raise HTTPException(status_code=404, detail="Avatar video not available")
        storage_path = video.avatar_video_path
    elif type == "subtitle":
        if not video.subtitle_path:
            raise HTTPException(status_code=404, detail="Subtitles not available")
        storage_path = video.subtitle_path
    else:
        raise HTTPException(status_code=400, detail="Invalid type")
    
    try:
        # Generate presigned URL
        url = await storage_service.get_presigned_url(storage_path, expires)
        
        return {
            "url": url,
            "expires_in": expires
        }
    except Exception as e:
        logger.error(f"Error generating presigned URL for video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating presigned URL")


@router.delete("/{video_id}")
async def delete_video(
    video_id: str = Path(..., description="Video ID"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Delete a video and all its associated files.
    
    Args:
        video_id: ID of the video to delete
    """
    # Initialize services
    video_service = VideoService(db)
    
    # Get video
    video = await video_service.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Check if user owns the video
    if video.user_id != current_user["sub"]:
        raise HTTPException(status_code=403, detail="Not authorized to delete this video")
    
    # Delete video
    success = await video_service.delete_video(video_id)
    
    if not success:
        raise HTTPException(status_code=500, detail="Error deleting video")
    
    return {"message": "Video deleted successfully"} 