from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import os
from datetime import datetime
import json
import uuid

from app.db.session import get_db
from app.models.transcription import Transcription, TranscriptionJob
from app.schemas.transcription import (
    TranscriptionRequest, 
    TranscriptionResponse, 
    TranscriptionUpdate,
    TranscriptionJobResponse
)
from app.services.whisper import extract_audio_from_video
from app.services.auth import get_current_user_id, verify_service_api_key
from app.core.config import settings

router = APIRouter()

@router.post("/", response_model=TranscriptionResponse)
async def create_transcription(
    request: TranscriptionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Create a new transcription request
    
    This endpoint accepts a transcription request and queues it for processing.
    It returns the created transcription object with a pending status.
    """
    # Ensure user has permission (user_id from token must match requested user_id)
    if user_id != request.user_id:
        raise HTTPException(
            status_code=403,
            detail="User ID in request does not match authenticated user"
        )
    
    # Extract file format from media URL
    file_format = os.path.splitext(request.media_url)[1].lower().lstrip('.')
    
    # Create transcription record
    new_transcription = Transcription(
        video_id=request.video_id,
        user_id=request.user_id,
        media_url=request.media_url,
        media_format=file_format,
        language=request.language or settings.DEFAULT_LANGUAGE,
        word_timestamps=request.word_timestamps,
        diarization=request.diarization,
        profanity_filter=request.profanity_filter,
        status="pending"
    )
    
    # Add to database
    db.add(new_transcription)
    db.commit()
    db.refresh(new_transcription)
    
    # Create transcription job
    job = TranscriptionJob(
        transcription_id=new_transcription.id,
        parameters={
            "language": request.language,
            "word_timestamps": request.word_timestamps,
            "diarization": request.diarization,
            "prompt": request.prompt,
            "profanity_filter": request.profanity_filter
        }
    )
    
    db.add(job)
    db.commit()
    
    # Queue job for processing (in a background task)
    from app.services.queue import queue_transcription_job
    background_tasks.add_task(queue_transcription_job, job.id)
    
    return new_transcription

@router.get("/{transcription_id}", response_model=TranscriptionResponse)
async def get_transcription(
    transcription_id: str = Path(..., description="ID of the transcription to retrieve"),
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Get a transcription by ID
    
    Returns the transcription details including status and results if available.
    """
    # Query the database
    transcription = db.query(Transcription).filter(
        Transcription.id == transcription_id
    ).first()
    
    # Check if transcription exists
    if not transcription:
        raise HTTPException(status_code=404, detail="Transcription not found")
    
    # Verify user has access to this transcription
    if transcription.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this transcription")
    
    return transcription

@router.get("/", response_model=List[TranscriptionResponse])
async def list_transcriptions(
    video_id: Optional[str] = Query(None, description="Filter by video ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    skip: int = Query(0, description="Number of records to skip"),
    limit: int = Query(100, description="Maximum number of records to return"),
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    List transcriptions
    
    Returns a list of transcriptions for the authenticated user.
    Optional filtering by video ID and status.
    """
    # Build query
    query = db.query(Transcription).filter(Transcription.user_id == user_id)
    
    # Apply filters
    if video_id:
        query = query.filter(Transcription.video_id == video_id)
    
    if status:
        query = query.filter(Transcription.status == status)
    
    # Apply pagination
    transcriptions = query.order_by(Transcription.created_at.desc()).offset(skip).limit(limit).all()
    
    return transcriptions

@router.patch("/{transcription_id}", response_model=TranscriptionResponse)
async def update_transcription(
    update_data: TranscriptionUpdate,
    transcription_id: str = Path(..., description="ID of the transcription to update"),
    db: Session = Depends(get_db),
    api_key: bool = Depends(verify_service_api_key)
):
    """
    Update a transcription (internal service endpoint)
    
    This endpoint is for internal service-to-service communication and requires an API key.
    It allows updating the status, progress, and results of a transcription.
    """
    # Query the database
    transcription = db.query(Transcription).filter(
        Transcription.id == transcription_id
    ).first()
    
    # Check if transcription exists
    if not transcription:
        raise HTTPException(status_code=404, detail="Transcription not found")
    
    # Update transcription fields
    update_dict = update_data.dict(exclude_unset=True)
    for key, value in update_dict.items():
        setattr(transcription, key, value)
    
    # If status is being updated to "completed" or "failed", set completed_at
    if update_dict.get("status") in ["completed", "failed"] and not transcription.completed_at:
        transcription.completed_at = datetime.now()
    
    # If status is being updated to "processing" and started_at is not set, set it
    if update_dict.get("status") == "processing" and not transcription.started_at:
        transcription.started_at = datetime.now()
    
    # Commit changes
    db.commit()
    db.refresh(transcription)
    
    # Notify Video Processing Service about the update
    if update_dict.get("status") in ["completed", "failed"]:
        # This would be done in a background task in a real implementation
        pass
    
    return transcription

@router.delete("/{transcription_id}", response_model=Dict[str, Any])
async def delete_transcription(
    transcription_id: str = Path(..., description="ID of the transcription to delete"),
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Delete a transcription
    
    Permanently removes a transcription and its associated files.
    """
    # Query the database
    transcription = db.query(Transcription).filter(
        Transcription.id == transcription_id
    ).first()
    
    # Check if transcription exists
    if not transcription:
        raise HTTPException(status_code=404, detail="Transcription not found")
    
    # Verify user has access to this transcription
    if transcription.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this transcription")
    
    # Delete associated files
    file_paths = [
        transcription.json_path,
        transcription.srt_path,
        transcription.vtt_path,
        transcription.txt_path
    ]
    
    for path in file_paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                # Log but continue if file deletion fails
                pass
    
    # Delete associated jobs
    jobs = db.query(TranscriptionJob).filter(
        TranscriptionJob.transcription_id == transcription_id
    ).all()
    
    for job in jobs:
        db.delete(job)
    
    # Delete the transcription
    db.delete(transcription)
    db.commit()
    
    return {"success": True, "message": "Transcription deleted"} 