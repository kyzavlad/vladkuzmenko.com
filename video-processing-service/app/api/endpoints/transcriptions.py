from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.db.postgres import get_db
from app.services.auth import get_current_user, verify_service_api_key
from app.services.transcription_service import TranscriptionService
from app.services.video_service import VideoService
from app.schemas.transcription import (
    TranscriptionRequest, TranscriptionResponse, 
    TranscriptionCallback, TranscriptionDetail
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/request/{video_id}", response_model=TranscriptionResponse)
async def request_transcription(
    video_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Request transcription for a video.
    """
    user_id = current_user["user_id"]
    
    # Get video service and validate access
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
    
    # Check if transcription is already in progress or completed
    if video.transcription_id:
        for task in video.tasks:
            if task.task_type == "transcription" and task.status in ["processing", "completed"]:
                return TranscriptionResponse(
                    video_id=video_id,
                    status="already_exists",
                    message=f"Transcription already {task.status} for this video",
                    transcription_id=video.transcription_id
                )
    
    # Create transcription service
    transcription_service = TranscriptionService(db)
    
    # Submit transcription request in background
    background_tasks.add_task(
        transcription_service.request_transcription,
        video_id=video_id,
        user_id=user_id
    )
    
    return TranscriptionResponse(
        video_id=video_id,
        status="processing",
        message="Transcription request submitted successfully."
    )


@router.get("/{video_id}", response_model=TranscriptionDetail)
async def get_transcription(
    video_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get the status and details of a video transcription.
    """
    user_id = current_user["user_id"]
    
    # Get video service and validate access
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
    
    # Check if transcription exists
    if not video.transcription_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No transcription found for this video"
        )
    
    # Get transcription details
    transcription_service = TranscriptionService(db)
    transcription_data = await transcription_service.get_transcription(video.transcription_id)
    
    if not transcription_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transcription not found in transcription service"
        )
    
    # Find transcription task
    transcription_task = None
    for task in video.tasks:
        if task.task_type == "transcription":
            transcription_task = task
            break
    
    return TranscriptionDetail(
        video_id=video_id,
        transcription_id=video.transcription_id,
        status=transcription_task.status if transcription_task else "unknown",
        progress=transcription_task.progress if transcription_task else 0,
        created_at=transcription_task.created_at if transcription_task else None,
        completed_at=transcription_task.completed_at if transcription_task else None,
        data=transcription_data
    )


@router.post("/callback", status_code=status.HTTP_200_OK)
async def transcription_callback(
    callback: TranscriptionCallback,
    db: AsyncSession = Depends(get_db),
    _: dict = Depends(verify_service_api_key)  # Ensure only authorized services can call this endpoint
):
    """
    Callback endpoint for the Transcription Service to update transcription status.
    This is called by the Transcription Service when a transcription job is completed.
    """
    transcription_id = callback.transcription_id
    status = callback.status
    
    logger.info(f"Received transcription callback for ID: {transcription_id}, status: {status}")
    
    # Update video and task records
    transcription_service = TranscriptionService(db)
    
    try:
        await transcription_service.update_transcription_status(
            transcription_id=transcription_id,
            status=status,
            result=callback.result,
            error=callback.error
        )
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error processing transcription callback: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process transcription callback"
        ) 