"""
Batch processing API endpoints for subtitle generation.
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.utils.batch_processor import BatchProcessor
from app.services.subtitles import SubtitleService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/batch", tags=["batch"])

# Initialize the batch processor
output_dir = os.path.join(settings.TEMP_DIRECTORY, "batch_outputs")
os.makedirs(output_dir, exist_ok=True)
batch_processor = BatchProcessor(output_dir=output_dir)

# Models for the API
class TranscriptSegment(BaseModel):
    """Transcript segment with timing information."""
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcript text")
    
class VideoTranscript(BaseModel):
    """Video with its transcript."""
    video_path: str = Field(..., description="Path to the video file")
    transcript: List[TranscriptSegment] = Field(..., description="Transcript segments")

class BatchRequest(BaseModel):
    """Request for batch subtitle generation."""
    videos: List[VideoTranscript] = Field(..., description="List of videos with transcripts")
    subtitle_formats: List[str] = Field(["srt", "vtt"], description="Subtitle formats to generate")
    generate_video: bool = Field(True, description="Whether to generate videos with burnt-in subtitles")
    job_id: Optional[str] = Field(None, description="Optional job ID")
    
    # Processing options
    video_quality: str = Field("medium", description="Video quality (low, medium, high)")
    optimize_positioning: bool = Field(False, description="Whether to optimize subtitle positioning")
    style_name: Optional[str] = Field(None, description="Subtitle style name")
    reading_speed_preset: Optional[str] = Field(None, description="Reading speed preset")
    detect_emphasis: bool = Field(False, description="Whether to detect emphasis")
    language: Optional[str] = Field(None, description="Subtitle language")
    auto_detect_language: bool = Field(False, description="Whether to auto-detect language")
    background_blur: float = Field(0.0, description="Background blur amount (0.0-1.0)")

class BatchJobResponse(BaseModel):
    """Response with batch job ID."""
    job_id: str = Field(..., description="Batch job ID")
    status: str = Field(..., description="Job status")
    total_videos: int = Field(..., description="Total number of videos")
    
class BatchStatusResponse(BaseModel):
    """Response with batch job status."""
    job_id: str = Field(..., description="Batch job ID")
    status: str = Field(..., description="Job status")
    total_videos: int = Field(..., description="Total number of videos")
    completed_videos: int = Field(..., description="Number of completed videos")
    failed_videos: int = Field(..., description="Number of failed videos")
    start_time: str = Field(..., description="Job start time")
    end_time: Optional[str] = Field(None, description="Job end time")
    progress: float = Field(..., description="Job progress (0.0-1.0)")
    current_video: Optional[str] = Field(None, description="Currently processing video")
    
class BatchListResponse(BaseModel):
    """Response with list of batch jobs."""
    jobs: List[BatchStatusResponse] = Field(..., description="List of batch jobs")

@router.post("/process", response_model=BatchJobResponse)
async def process_batch(
    request: BatchRequest,
    background_tasks: BackgroundTasks
) -> BatchJobResponse:
    """
    Process a batch of videos, generating subtitles in various formats.
    
    Returns a job ID for tracking the batch processing.
    """
    try:
        # Extract video paths and transcripts
        video_paths = []
        transcripts = []
        
        for video in request.videos:
            video_paths.append(video.video_path)
            transcript = [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text
                }
                for segment in video.transcript
            ]
            transcripts.append(transcript)
        
        # Prepare batch options
        batch_options = {
            "video_quality": request.video_quality,
            "optimize_positioning": request.optimize_positioning,
            "style_name": request.style_name,
            "reading_speed_preset": request.reading_speed_preset,
            "detect_emphasis": request.detect_emphasis,
            "language": request.language,
            "auto_detect_language": request.auto_detect_language,
            "background_blur": request.background_blur
        }
        
        # Start batch processing
        job_id = await batch_processor.process_batch(
            video_paths=video_paths,
            transcripts=transcripts,
            job_id=request.job_id,
            subtitle_formats=request.subtitle_formats,
            generate_video=request.generate_video,
            batch_options=batch_options
        )
        
        # Get job status
        job_status = batch_processor.get_job_status(job_id)
        
        return BatchJobResponse(
            job_id=job_id,
            status=job_status["status"],
            total_videos=job_status["total_videos"]
        )
    
    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing batch: {str(e)}")

@router.get("/status/{job_id}", response_model=BatchStatusResponse)
async def get_batch_status(job_id: str) -> BatchStatusResponse:
    """
    Get the status of a batch job.
    """
    try:
        job_status = batch_processor.get_job_status(job_id)
        
        # Calculate progress
        total = job_status["total_videos"]
        completed = job_status["completed_videos"]
        failed = job_status["failed_videos"]
        
        progress = 0.0
        if total > 0:
            progress = (completed + failed) / total
        
        return BatchStatusResponse(
            job_id=job_id,
            status=job_status["status"],
            total_videos=job_status["total_videos"],
            completed_videos=job_status["completed_videos"],
            failed_videos=job_status["failed_videos"],
            start_time=job_status["start_time"],
            end_time=job_status["end_time"],
            progress=progress,
            current_video=job_status.get("current_video")
        )
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    except Exception as e:
        logger.error(f"Error getting batch status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting batch status: {str(e)}")

@router.get("/list", response_model=BatchListResponse)
async def list_batch_jobs() -> BatchListResponse:
    """
    List all batch jobs.
    """
    try:
        jobs = batch_processor.list_jobs()
        
        job_statuses = []
        for job in jobs:
            total = job["total_videos"]
            completed = job["completed_videos"]
            failed = job["failed_videos"]
            
            progress = 0.0
            if total > 0:
                progress = (completed + failed) / total
            
            job_statuses.append(
                BatchStatusResponse(
                    job_id=job["job_id"],
                    status=job["status"],
                    total_videos=job["total_videos"],
                    completed_videos=job["completed_videos"],
                    failed_videos=job["failed_videos"],
                    start_time=job["start_time"],
                    end_time=job["end_time"],
                    progress=progress,
                    current_video=job.get("current_video")
                )
            )
        
        return BatchListResponse(jobs=job_statuses)
    
    except Exception as e:
        logger.error(f"Error listing batch jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing batch jobs: {str(e)}")

@router.delete("/job/{job_id}", response_model=Dict[str, str])
async def delete_batch_job(job_id: str) -> Dict[str, str]:
    """
    Delete a batch job and its output files.
    """
    try:
        # Check if job exists
        job_status = batch_processor.get_job_status(job_id)
        
        # Remove job from processor
        batch_processor.jobs.pop(job_id, None)
        
        # Schedule background task to delete output directory
        job_dir = os.path.join(output_dir, job_id)
        if os.path.exists(job_dir):
            # Use a background task to delete the directory
            background_tasks = BackgroundTasks()
            async def delete_directory():
                try:
                    import shutil
                    shutil.rmtree(job_dir)
                except Exception as e:
                    logger.error(f"Error deleting job directory {job_dir}: {str(e)}")
                    
            background_tasks.add_task(delete_directory)
        
        return {"status": "deleted", "job_id": job_id}
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    except Exception as e:
        logger.error(f"Error deleting batch job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting batch job: {str(e)}")

@router.get("/job/{job_id}/summary", response_model=Dict[str, Any])
async def get_batch_summary(job_id: str) -> Dict[str, Any]:
    """
    Get the complete summary of a batch job, including all results.
    """
    try:
        # Check if job exists
        job_status = batch_processor.get_job_status(job_id)
        
        # Get summary file path
        job_dir = os.path.join(output_dir, job_id)
        summary_path = os.path.join(job_dir, "batch_summary.json")
        
        # Check if summary file exists
        if not os.path.exists(summary_path):
            # Return current job status if summary file not found
            return job_status
        
        # Read summary file
        with open(summary_path, 'r') as f:
            import json
            summary = json.load(f)
            
        return summary
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    except Exception as e:
        logger.error(f"Error getting batch summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting batch summary: {str(e)}") 