"""
Subtitle generation API endpoints.
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, File, UploadFile, Form, BackgroundTasks, HTTPException, Depends
from pydantic import BaseModel, Field

from app.services.subtitles import SubtitleService, SubtitleFormat
from app.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize the subtitle service
subtitle_service = SubtitleService()

class TranscriptSegment(BaseModel):
    """Segment in a transcript."""
    start: float
    end: float
    text: str
    style: Optional[Dict[str, Any]] = None

class Transcript(BaseModel):
    """Full transcript schema."""
    segments: List[TranscriptSegment]
    metadata: Optional[Dict[str, Any]] = None

class SubtitleRequest(BaseModel):
    """Request for subtitle generation."""
    transcript: Transcript
    format: str = "srt"
    style_name: Optional[str] = None
    custom_style: Optional[Dict[str, Any]] = None
    reading_speed_preset: Optional[str] = None
    adjust_timing: bool = False
    detect_emphasis: bool = False
    language: Optional[str] = None
    auto_detect_language: bool = False

class MultipleOutputRequest(BaseModel):
    """Request for generating multiple subtitle outputs."""
    transcript: Transcript
    subtitle_formats: List[str] = Field(default=["srt", "vtt"])
    generate_video: bool = True
    video_quality: Optional[str] = "medium"
    optimize_positioning: bool = False
    style_name: Optional[str] = None
    custom_style: Optional[Dict[str, Any]] = None
    reading_speed_preset: Optional[str] = None
    detect_emphasis: bool = False
    language: Optional[str] = None
    auto_detect_language: bool = False
    background_blur: float = 0.0
    
class SubtitleResponse(BaseModel):
    """Response for subtitle generation."""
    subtitle_path: str
    format: str

class MultipleOutputResponse(BaseModel):
    """Response for multiple subtitle output generation."""
    output_files: Dict[str, str]
    manifest_path: str

def get_temp_path(filename: str) -> str:
    """Get a temporary file path."""
    os.makedirs(settings.TEMP_DIRECTORY, exist_ok=True)
    return os.path.join(settings.TEMP_DIRECTORY, filename)

def validate_format(format_name: str) -> SubtitleFormat:
    """Validate the subtitle format."""
    try:
        return SubtitleFormat(format_name.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unsupported subtitle format: {format_name}")

@router.post("/generate", response_model=SubtitleResponse)
async def generate_subtitle(request: SubtitleRequest):
    """
    Generate a subtitle file from a transcript.
    """
    # Convert transcript from Pydantic model to dict
    transcript_dict = {"segments": [dict(segment) for segment in request.transcript.segments]}
    if request.transcript.metadata:
        transcript_dict["metadata"] = request.transcript.metadata
    
    # Validate format
    format_enum = validate_format(request.format)
    
    # Generate a temporary output path
    output_path = get_temp_path(f"subtitle_{id(request)}_{request.format}")
    
    try:
        # Generate the subtitle file
        subtitle_path = await subtitle_service.generate_subtitles(
            transcript=transcript_dict,
            output_path=output_path,
            format=format_enum,
            style_name=request.style_name,
            custom_style=request.custom_style,
            reading_speed_preset=request.reading_speed_preset,
            adjust_timing=request.adjust_timing,
            detect_emphasis=request.detect_emphasis,
            language=request.language,
            auto_detect_language=request.auto_detect_language
        )
        
        return {"subtitle_path": subtitle_path, "format": request.format}
        
    except Exception as e:
        logger.error(f"Error generating subtitle: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating subtitle: {str(e)}")

@router.post("/multiple-outputs", response_model=MultipleOutputResponse)
async def generate_multiple_outputs(
    request: MultipleOutputRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate multiple subtitle outputs (subtitle files and video with burnt-in subtitles).
    
    This endpoint generates both subtitle files in multiple formats and a video with 
    burnt-in subtitles, packaging everything together with a manifest.
    """
    # Convert transcript from Pydantic model to dict
    transcript_dict = {"segments": [dict(segment) for segment in request.transcript.segments]}
    if request.transcript.metadata:
        transcript_dict["metadata"] = request.transcript.metadata
    
    # Create a unique output directory
    output_dir = get_temp_path(f"multiple_output_{id(request)}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Validate all formats
    valid_formats = []
    for format_name in request.subtitle_formats:
        try:
            validate_format(format_name)
            valid_formats.append(format_name)
        except HTTPException:
            # Just skip invalid formats and log a warning
            logger.warning(f"Skipping unsupported subtitle format: {format_name}")
    
    if not valid_formats:
        raise HTTPException(status_code=400, detail="No valid subtitle formats provided")
    
    try:
        # Generate multiple outputs
        result_files = await subtitle_service.generate_multiple_outputs(
            video_path=settings.DEFAULT_VIDEO_PATH,  # This would come from file upload in a real scenario
            transcript=transcript_dict,
            output_dir=output_dir,
            subtitle_formats=valid_formats,
            generate_video=request.generate_video,
            video_quality=request.video_quality,
            optimize_positioning=request.optimize_positioning,
            style_name=request.style_name,
            custom_style=request.custom_style,
            reading_speed_preset=request.reading_speed_preset,
            detect_emphasis=request.detect_emphasis,
            language=request.language,
            auto_detect_language=request.auto_detect_language,
            background_blur=request.background_blur
        )
        
        # Schedule cleanup of temporary files after a delay (e.g., 1 hour)
        background_tasks.add_task(
            cleanup_temp_files, 
            output_dir, 
            delay_seconds=3600
        )
        
        return {
            "output_files": result_files,
            "manifest_path": result_files.get("manifest", "")
        }
        
    except Exception as e:
        logger.error(f"Error generating multiple outputs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating multiple outputs: {str(e)}")

async def cleanup_temp_files(directory: str, delay_seconds: int = 3600):
    """
    Clean up temporary files after a delay.
    
    Args:
        directory: Directory to clean up
        delay_seconds: Delay before cleanup (in seconds)
    """
    await asyncio.sleep(delay_seconds)
    
    try:
        import shutil
        if os.path.exists(directory) and os.path.isdir(directory):
            shutil.rmtree(directory)
            logger.info(f"Cleaned up temporary directory: {directory}")
    except Exception as e:
        logger.error(f"Error cleaning up temporary directory: {str(e)}")

@router.post("/upload-video")
async def upload_video(
    video: UploadFile = File(...),
):
    """
    Upload a video file for subtitle processing.
    
    This endpoint allows uploading a video file which can then be used
    with other endpoints for subtitle generation.
    """
    # Create a unique filename
    filename = f"uploaded_{id(video)}_{video.filename}"
    file_path = get_temp_path(filename)
    
    # Save the uploaded file
    with open(file_path, "wb") as f:
        contents = await video.read()
        f.write(contents)
    
    return {"filename": filename, "path": file_path} 