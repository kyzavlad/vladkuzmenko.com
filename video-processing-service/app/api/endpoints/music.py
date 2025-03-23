"""
Music Service API

API endpoints for the music selection, analysis, and volume adjustment services.
"""

import os
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, File, UploadFile, Form, Body
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.music import VolumeAdjuster, MusicSelector, BPMDetector

# Setup router
router = APIRouter()
logger = logging.getLogger(__name__)

class DynamicVolumeRequest(BaseModel):
    video_path: str = Field(..., description="Path to the input video file")
    music_path: str = Field(..., description="Path to the music file")
    output_path: str = Field(..., description="Path to save the output file")
    speech_volume: float = Field(1.0, description="Volume level for speech (0-1)")
    music_volume: float = Field(0.2, description="Base volume level for music (0-1)")
    ducking_ratio: float = Field(0.25, description="How much to reduce music during speech (0-1)")
    music_start_time: float = Field(0.0, description="Start time for music in seconds")
    music_fade_in: float = Field(2.0, description="Fade-in duration for music in seconds")
    music_fade_out: float = Field(3.0, description="Fade-out duration for music in seconds")

class DynamicVolumeResponse(BaseModel):
    status: str
    task_id: Optional[str] = None
    output_path: str
    detected_segments: Optional[List[Dict[str, Any]]] = None
    duration: Optional[float] = None
    error: Optional[str] = None

@router.post("/adjust-volume", response_model=DynamicVolumeResponse)
async def adjust_volume(
    request: DynamicVolumeRequest,
    background_tasks: BackgroundTasks
):
    """
    Apply dynamic volume adjustment to a video, automatically lowering music volume during speech.
    """
    # Validate input/output paths
    for path_field in ["video_path", "music_path", "output_path"]:
        path = getattr(request, path_field)
        if not os.path.isabs(path):
            # Resolve relative to the output directory
            setattr(request, path_field, os.path.join(settings.OUTPUT_DIR, path))
    
    video_path = request.video_path
    music_path = request.music_path
    output_path = request.output_path
    
    # Check that video and music files exist
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video file not found: {video_path}")
    
    if not os.path.exists(music_path):
        raise HTTPException(status_code=404, detail=f"Music file not found: {music_path}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Set up volume adjuster configuration
    config = {
        "ffmpeg_path": settings.FFMPEG_PATH,
        "ffprobe_path": settings.FFPROBE_PATH,
        "temp_dir": settings.TEMP_DIR,
        "speech_threshold": -25,  # dB
        "music_normal_volume": request.music_volume,
        "music_ducked_volume": request.music_volume * request.ducking_ratio,
        "attack_time": 0.3,  # seconds
        "release_time": 0.5  # seconds
    }
    
    # Create volume adjuster
    volume_adjuster = VolumeAdjuster(config)
    
    # Process in background
    task_id = f"volume_adjust_{os.path.basename(video_path)}_{os.path.basename(music_path)}"
    
    # Initial response
    response = DynamicVolumeResponse(
        status="processing",
        task_id=task_id,
        output_path=output_path
    )
    
    # Function to run in background
    def process_volume_adjustment():
        try:
            # Duck music in video
            result = volume_adjuster.duck_music_in_video(
                video_path=video_path,
                music_path=music_path,
                output_path=output_path,
                speech_volume=request.speech_volume,
                music_volume=request.music_volume,
                ducking_ratio=request.ducking_ratio,
                music_start_time=request.music_start_time,
                music_fade_in=request.music_fade_in,
                music_fade_out=request.music_fade_out
            )
            
            logger.info(f"Volume adjustment complete: {output_path}")
        except Exception as e:
            logger.error(f"Error during volume adjustment: {str(e)}")
    
    # Add task to background tasks
    background_tasks.add_task(process_volume_adjustment)
    
    return response

@router.get("/volume-adjust-status/{task_id}")
async def get_volume_adjust_status(task_id: str):
    """
    Get the status of a volume adjustment task.
    """
    # In a real implementation, this would check a database or cache
    # for the task status. For now, we'll just return a placeholder.
    return {
        "status": "completed",
        "task_id": task_id,
        "message": "This is a placeholder. In a real implementation, this would return the actual task status."
    }

# Future endpoint placeholder for music recommendations
@router.post("/recommend-music")
async def recommend_music(
    video_path: str = Body(..., embed=True),
    duration: Optional[float] = Body(None, embed=True),
    mood: Optional[str] = Body(None, embed=True),
    genre: Optional[str] = Body(None, embed=True)
):
    """
    Recommend music tracks based on video content analysis.
    """
    # Placeholder for future implementation
    return {
        "status": "not_implemented",
        "message": "This endpoint will be implemented in a future update."
    }

class BPMDetectionRequest(BaseModel):
    audio_path: str = Field(..., description="Path to the input audio file")
    start_time: float = Field(0.0, description="Start time for analysis in seconds")
    duration: Optional[float] = Field(None, description="Duration to analyze in seconds (None for default)")

class BPMDetectionResponse(BaseModel):
    status: str
    bpm: Optional[float] = None
    confidence: Optional[float] = None
    file_path: str
    rhythm_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@router.post("/detect-bpm", response_model=BPMDetectionResponse)
async def detect_bpm(request: BPMDetectionRequest):
    """
    Detect BPM (tempo) and rhythm information in an audio file.
    """
    # Validate input path
    audio_path = request.audio_path
    if not os.path.isabs(audio_path):
        # Resolve relative to the output directory
        audio_path = os.path.join(settings.OUTPUT_DIR, audio_path)
    
    # Check that audio file exists
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail=f"Audio file not found: {audio_path}")
    
    # Set up BPM detector configuration
    config = {
        "ffmpeg_path": settings.FFMPEG_PATH,
        "ffprobe_path": settings.FFPROBE_PATH,
        "temp_dir": settings.TEMP_DIR
    }
    
    # Create BPM detector
    bpm_detector = BPMDetector(config)
    
    # Detect BPM
    result = bpm_detector.detect_bpm(
        audio_path=audio_path,
        start_time=request.start_time,
        duration=request.duration
    )
    
    # Convert to response format
    response = BPMDetectionResponse(
        status=result.get("status", "error"),
        bpm=result.get("bpm"),
        confidence=result.get("confidence"),
        file_path=audio_path,
        rhythm_info=result.get("rhythm_info"),
        error=result.get("error")
    )
    
    return response

class BPMMatchRequest(BaseModel):
    target_bpm: float = Field(..., description="Target BPM to match")
    bpm_range: float = Field(5.0, description="Acceptable BPM range around the target")
    audio_directory: str = Field(..., description="Directory path containing audio files to analyze")

class BPMMatchResponse(BaseModel):
    status: str
    matching_tracks: List[Dict[str, Any]] = []
    error: Optional[str] = None

@router.post("/match-bpm", response_model=BPMMatchResponse)
async def match_bpm(
    request: BPMMatchRequest,
    background_tasks: BackgroundTasks
):
    """
    Find audio tracks matching a target BPM within a specified range.
    """
    # Validate input directory
    audio_directory = request.audio_directory
    if not os.path.isabs(audio_directory):
        # Resolve relative to the output directory
        audio_directory = os.path.join(settings.OUTPUT_DIR, audio_directory)
    
    # Check that directory exists
    if not os.path.exists(audio_directory) or not os.path.isdir(audio_directory):
        raise HTTPException(status_code=404, detail=f"Directory not found: {audio_directory}")
    
    # Set up BPM detector configuration
    config = {
        "ffmpeg_path": settings.FFMPEG_PATH,
        "ffprobe_path": settings.FFPROBE_PATH,
        "temp_dir": settings.TEMP_DIR
    }
    
    # Create BPM detector
    bpm_detector = BPMDetector(config)
    
    # Task ID
    task_id = f"bpm_match_{os.path.basename(audio_directory)}_{request.target_bpm}"
    
    # Initial response
    response = BPMMatchResponse(
        status="processing",
        matching_tracks=[]
    )
    
    # Function to run in background
    def process_bpm_matching():
        try:
            # Audio file extensions
            audio_extensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']
            
            # List of tracks with BPM information
            tracks = []
            
            # Process each audio file
            for root, _, files in os.walk(audio_directory):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in audio_extensions):
                        file_path = os.path.join(root, file)
                        
                        # Detect BPM
                        result = bpm_detector.detect_bpm(file_path)
                        
                        if result["status"] == "success":
                            tracks.append({
                                "file_path": file_path,
                                "bpm": result["bpm"],
                                "confidence": result.get("confidence", 0),
                                "rhythm_info": result.get("rhythm_info", {})
                            })
            
            # Match tracks to target BPM
            matched_tracks = bpm_detector.match_bpm(
                request.target_bpm, 
                request.bpm_range, 
                tracks
            )
            
            # Store results (in a real implementation, this would update a database)
            logger.info(f"BPM matching completed for {task_id}: found {len(matched_tracks)} matches")
            
            # In a real implementation, you would store this result somewhere
            # for retrieval by the status endpoint
            
        except Exception as e:
            logger.error(f"Error during BPM matching: {str(e)}")
    
    # Add task to background tasks
    background_tasks.add_task(process_bpm_matching)
    
    return response 