"""
Audio Enhancement API Endpoints

This module provides REST API endpoints for the Audio Enhancement Suite.
"""

import os
import logging
import json
import asyncio
import shutil
import tempfile
import time
from typing import Dict, List, Any, Optional
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, Form, BackgroundTasks, HTTPException, Depends, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app.services.audio.audio_enhancer import AudioEnhancer
from app.services.audio.dynamics_processor import DynamicsProcessor
from app.services.music.genre_classifier import GenreClassifier
from app.services.music.bpm_detector import BPMDetector
from app.services.music.volume_adjuster import VolumeAdjuster
from app.services.music.mood_analyzer import MoodAnalyzer
from app.services.music.emotional_arc_mapper import EmotionalArcMapper
from app.services.music.music_library import MusicLibrary
from app.services.music.audio_fingerprinter import AudioFingerprinter
from app.services.sound_effects.sound_effects_library import SoundEffectsLibrary
from app.services.sound_effects.sound_effects_processor import SoundEffectsProcessor
from app.services.sound_effects.sound_effects_recommender import SoundEffectsRecommender
from app.core.config import settings
from app.core.auth import get_current_user, User

router = APIRouter()
logger = logging.getLogger(__name__)

# Pydantic models for API
class NoiseReductionOptions(BaseModel):
    noise_profile_id: Optional[str] = None
    auto_detect: bool = True
    strength: Optional[float] = Field(None, ge=0.0, le=1.0)
    noise_sample: Optional[Dict[str, Any]] = None

class VoiceEnhancementOptions(BaseModel):
    apply_eq: bool = True
    apply_compression: bool = True
    apply_de_essing: bool = True
    apply_harmonic_enhancement: bool = True
    male_voice_boost: bool = False
    female_voice_boost: bool = False
    clarity: Optional[float] = Field(None, ge=0.0, le=1.0)
    warmth: Optional[float] = Field(None, ge=0.0, le=1.0)

class DynamicsProcessingOptions(BaseModel):
    apply_compression: bool = True
    apply_limiting: bool = True
    apply_expansion: bool = False
    apply_gating: bool = False
    target_loudness: Optional[float] = Field(None, ge=-30.0, le=-8.0, description="Target loudness in LUFS, broadcast standard is -14")
    dry_wet_mix: float = Field(1.0, ge=0.0, le=1.0, description="Mix between original and processed audio")
    preset: Optional[str] = Field(None, description="Preset name (voice_broadcast, voice_intimate, music_master, dialog_leveler, transparent)")

class AudioEnhancementRequest(BaseModel):
    apply_noise_reduction: bool = True
    apply_voice_enhancement: bool = False
    apply_dynamics_processing: bool = False
    noise_reduction: Optional[NoiseReductionOptions] = None
    voice_enhancement: Optional[VoiceEnhancementOptions] = None
    dynamics_processing: Optional[DynamicsProcessingOptions] = None
    
class AudioJobResponse(BaseModel):
    job_id: str
    status: str
    input_path: str
    output_path: Optional[str] = None
    created_at: float

# For dynamics processing endpoint
class DynamicsProcessRequest(BaseModel):
    preset: Optional[str] = Field(None, description="Preset name (voice_broadcast, voice_intimate, music_master, dialog_leveler, transparent)")
    target_loudness: Optional[float] = Field(-14.0, ge=-30.0, le=-8.0, description="Target loudness in LUFS")
    comp_threshold: Optional[float] = Field(None, le=0.0, description="Compression threshold in dB")
    comp_ratio: Optional[float] = Field(None, ge=1.0, description="Compression ratio")
    comp_attack: Optional[float] = Field(None, ge=0.001, description="Compressor attack time in seconds")
    comp_release: Optional[float] = Field(None, ge=0.001, description="Compressor release time in seconds")
    apply_compression: bool = True
    apply_limiting: bool = True
    apply_expansion: bool = False
    apply_gating: bool = False
    dry_wet_mix: float = Field(1.0, ge=0.0, le=1.0, description="Mix between original and processed audio")

class GenreClassificationRequest(BaseModel):
    """Request model for genre classification"""
    top_n: int = Field(3, ge=1, le=10, description="Number of top genres to return")

class GenreRecommendationRequest(BaseModel):
    """Request model for genre recommendations"""
    video_mood: str = Field(..., description="Mood of the video content (e.g., happy, sad, energetic)")
    video_genre: str = Field(..., description="Genre of the video content (e.g., documentary, tutorial)")
    top_n: int = Field(5, ge=1, le=10, description="Number of top recommendations to return")

class BPMDetectionRequest(BaseModel):
    """Request model for BPM detection"""
    file_path: str = Field(..., description="Path to the audio or video file")

class BPMMatchingRequest(BaseModel):
    """Request model for BPM matching"""
    target_bpm: float = Field(..., ge=40.0, le=240.0, description="Target BPM to match")
    file_paths: List[str] = Field(..., description="List of paths to audio files to analyze")
    tolerance: float = Field(5.0, ge=1.0, le=20.0, description="Allowed BPM difference for matching")
    match_style: str = Field("exact", description="Matching style: exact, double, or harmonic")

class BPMSuggestionRequest(BaseModel):
    """Request model for content-based BPM suggestion"""
    content_type: str = Field(..., description="Type of content (e.g., interview, documentary, action)")

# For music volume adjustment endpoint
class VolumeAdjustmentRequest(BaseModel):
    """Request model for dynamic volume adjustment of music during speech."""
    video_path: str = Field(..., description="Path to the video file")
    music_path: str = Field(..., description="Path to the music file")
    music_start_time: float = Field(0.0, ge=0.0, description="Start time for music in seconds")
    music_end_time: Optional[float] = Field(None, description="End time for music in seconds (optional)")
    default_volume: float = Field(0.7, ge=0.0, le=1.0, description="Default music volume")
    ducking_amount: float = Field(0.3, ge=0.0, le=1.0, description="Volume level during speech")
    fade_in_time: float = Field(0.5, ge=0.1, le=3.0, description="Fade-in time after speech (seconds)")
    fade_out_time: float = Field(0.8, ge=0.1, le=3.0, description="Fade-out time before speech (seconds)")
    keep_original_audio: bool = Field(True, description="Keep original audio from video")

# In-memory storage for job status (in a real app, use a database)
audio_jobs = {}

def get_temp_path(prefix: str) -> str:
    """Get a temporary path."""
    # Create a temporary directory within the configured temp path
    os.makedirs(settings.temp_path, exist_ok=True)
    temp_dir = os.path.join(settings.temp_path, f"{prefix}_{int(time.time())}")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

def get_enhancer_config() -> dict:
    """Get configuration for the Audio Enhancer."""
    return {
        'temp_dir': settings.temp_path,
        'ffmpeg_path': settings.ffmpeg_path,
        'ffprobe_path': settings.ffprobe_path,
        'noise_reduction': {
            'reduction_strength': 0.75,
            'n_fft': 2048,
            'hop_length': 512,
            'n_std_thresh': 1.5,
            'freq_mask_smooth_hz': 500,
            'time_mask_smooth_ms': 50,
            'chunk_size': 60,
            'padding': 1
        },
        'voice_enhancement': {
            'eq_presence_gain': 3.0,
            'de_essing_strength': 0.5,
            'clarity': 0.4
        },
        'dynamics_processing': {
            'comp_threshold': -24.0,
            'comp_ratio': 2.0,
            'limit_threshold': -1.0,
            'auto_makeup_gain': True
        }
    }

async def process_audio_job(job_id: str, file_path: str, request_data: dict, output_dir: str):
    """
    Process an audio enhancement job in the background.
    
    Args:
        job_id: Unique job identifier
        file_path: Path to the audio or video file
        request_data: Enhancement request data
        output_dir: Directory to save outputs
    """
    try:
        # Update job status
        audio_jobs[job_id]['status'] = 'processing'
        
        # Initialize Audio Enhancer
        config = get_enhancer_config()
        enhancer = AudioEnhancer(config)
        
        # Determine if it's a video or audio file
        is_video = os.path.splitext(file_path)[1].lower() in [
            '.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv'
        ]
        
        # Generate output path
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}_enhanced{ext}")
        
        # Process options
        options = {
            'apply_noise_reduction': request_data.get('apply_noise_reduction', True),
            'noise_reduction': request_data.get('noise_reduction', {})
        }
        
        # Process the file
        if is_video:
            result = await enhancer.enhance_video_audio(
                video_path=file_path,
                output_path=output_path,
                options=options
            )
        else:
            result = await enhancer.enhance_audio(
                input_path=file_path,
                output_path=output_path,
                options=options
            )
        
        # Update job with results
        if "error" in result:
            audio_jobs[job_id].update({
                'status': 'failed',
                'error': result["error"],
                'completed_at': time.time()
            })
        else:
            audio_jobs[job_id].update({
                'status': 'completed',
                'output_path': result["output_path"],
                'processing_steps': result.get("processing_steps", []),
                'completed_at': time.time()
            })
        
    except Exception as e:
        logger.error(f"Error processing audio job {job_id}: {str(e)}")
        audio_jobs[job_id].update({
            'status': 'failed',
            'error': str(e),
            'completed_at': time.time()
        })
        
        # Clean up
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)

@router.post("/upload", response_class=JSONResponse)
async def upload_audio(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """
    Upload an audio or video file for enhancement.
    
    Returns:
        JSON response with the path to the uploaded file
    """
    # Create a temporary path for the upload
    temp_dir = get_temp_path("upload")
    file_path = os.path.join(temp_dir, file.filename)
    
    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {
        "filename": file.filename,
        "file_path": file_path
    }

@router.post("/enhance", response_model=AudioJobResponse)
async def enhance_audio(
    background_tasks: BackgroundTasks,
    request: AudioEnhancementRequest,
    file_path: str = Query(..., description="Path to the uploaded audio or video file"),
    current_user: User = Depends(get_current_user)
):
    """
    Enhance audio quality of a file.
    
    Args:
        request: Audio enhancement request
        file_path: Path to the uploaded audio or video file
        
    Returns:
        Job status information
    """
    # Validate file path
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Create output directory
    output_dir = get_temp_path("audio_enhanced")
    
    # Create a unique job ID
    job_id = f"audio_{int(time.time())}_{os.path.basename(file_path)}"
    
    # Store job info
    audio_jobs[job_id] = {
        'job_id': job_id,
        'status': 'queued',
        'input_path': file_path,
        'output_dir': output_dir,
        'created_at': time.time()
    }
    
    # Start processing in the background
    request_data = request.dict()
    background_tasks.add_task(process_audio_job, job_id, file_path, request_data, output_dir)
    
    return audio_jobs[job_id]

@router.get("/jobs/{job_id}", response_class=JSONResponse)
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get the status of an audio enhancement job.
    
    Args:
        job_id: Job identifier
        
    Returns:
        Job status information
    """
    if job_id not in audio_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return audio_jobs[job_id]

@router.get("/jobs/{job_id}/download", response_class=FileResponse)
async def download_enhanced_file(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Download the enhanced audio or video file.
    
    Args:
        job_id: Job identifier
        
    Returns:
        Enhanced file
    """
    if job_id not in audio_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = audio_jobs[job_id]
    
    if job.get('status') != 'completed':
        raise HTTPException(status_code=400, detail="Job not completed")
    
    output_path = job.get('output_path')
    
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Enhanced file not found")
    
    return FileResponse(output_path)

@router.post("/analyze", response_class=JSONResponse)
async def analyze_audio(
    file_path: str = Query(..., description="Path to the audio or video file to analyze"),
    current_user: User = Depends(get_current_user)
):
    """
    Analyze audio to identify characteristics and potential issues.
    
    Args:
        file_path: Path to the audio or video file
        
    Returns:
        Analysis results
    """
    # Validate file path
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Initialize Audio Enhancer
    config = get_enhancer_config()
    enhancer = AudioEnhancer(config)
    
    # Determine if it's a video or audio file
    is_video = os.path.splitext(file_path)[1].lower() in [
        '.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv'
    ]
    
    # If it's a video, extract audio first
    if is_video:
        # Extract audio to a temporary file
        temp_dir = get_temp_path("audio_analysis")
        audio_path = os.path.join(temp_dir, f"{os.path.basename(file_path)}_audio.wav")
        
        # Run ffmpeg to extract audio
        ffmpeg_cmd = [
            settings.ffmpeg_path,
            "-i", file_path,
            "-q:a", "0",
            "-map", "a",
            "-vn",
            audio_path
        ]
        
        extract_process = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await extract_process.communicate()
        
        if extract_process.returncode != 0:
            logger.error(f"Failed to extract audio: {stderr.decode()}")
            raise HTTPException(status_code=500, detail="Failed to extract audio from video")
        
        # Analyze the extracted audio
        result = await enhancer.analyze_audio(audio_path)
        
        # Clean up
        if os.path.exists(audio_path):
            os.remove(audio_path)
    else:
        # Analyze audio file directly
        result = await enhancer.analyze_audio(file_path)
    
    return result

@router.get("/enhancements", response_class=JSONResponse)
async def get_available_enhancements(
    current_user: User = Depends(get_current_user)
):
    """
    Get a list of available audio enhancement features.
    
    Returns:
        Dictionary with enhancement names and availability status
    """
    # Initialize Audio Enhancer
    config = get_enhancer_config()
    enhancer = AudioEnhancer(config)
    
    # Get available enhancements
    enhancements = enhancer.get_available_enhancements()
    
    # Add music and genre classification features
    enhancements["features"]["music_analysis"] = {
        "available": True,
        "features": {
            "genre_classification": {
                "available": True,
                "description": "Classify audio into musical genres"
            },
            "genre_recommendation": {
                "available": True,
                "description": "Recommend music genres based on video content"
            },
            "bpm_detection": {
                "available": True,
                "description": "Detect and analyze tempo (BPM) in audio"
            },
            "bpm_matching": {
                "available": True,
                "description": "Match music tracks based on tempo"
            },
            "content_tempo_suggestion": {
                "available": True,
                "description": "Suggest appropriate tempos for different content types"
            },
            "dynamic_volume_adjustment": {
                "available": True,
                "description": "Automatically adjust music volume during speech segments"
            },
            "content_mood_analysis": {
                "available": True,
                "description": "Analyze the emotional mood of video content"
            },
            "emotional_arc_mapping": {
                "available": True,
                "description": "Map the emotional arc of video content over time"
            }
        }
    }
    
    # Add endpoints for music analysis
    if "endpoints" in enhancements:
        enhancements["endpoints"].extend([
            {
                "path": "/audio/classify-genre",
                "method": "POST",
                "description": "Classify the genre of an audio file"
            },
            {
                "path": "/audio/recommend-genres",
                "method": "POST",
                "description": "Get music genre recommendations for video content"
            },
            {
                "path": "/audio/detect-bpm",
                "method": "POST",
                "description": "Detect the BPM (tempo) of an audio file"
            },
            {
                "path": "/audio/match-bpm",
                "method": "POST",
                "description": "Find tracks that match a target BPM"
            },
            {
                "path": "/audio/suggest-bpm",
                "method": "POST",
                "description": "Suggest appropriate BPM for different content types"
            },
            {
                "path": "/audio/adjust-music-volume",
                "method": "POST",
                "description": "Dynamically adjust music volume during speech segments"
            },
            {
                "path": "/audio/analyze-mood",
                "method": "POST",
                "description": "Analyze the emotional mood of video content"
            },
            {
                "path": "/audio/map-emotional-arc",
                "method": "POST",
                "description": "Map the emotional arc of video content over time"
            }
        ])
    
    return enhancements

@router.post("/batch", response_class=JSONResponse)
async def batch_process(
    background_tasks: BackgroundTasks,
    file_paths: List[str],
    request: AudioEnhancementRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Process multiple audio or video files in batch.
    
    Args:
        file_paths: List of paths to audio or video files
        request: Audio enhancement request
        
    Returns:
        Batch job status information
    """
    # Validate file paths
    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    
    # Create output directory
    output_dir = get_temp_path("audio_batch")
    
    # Create a unique job ID
    job_id = f"audio_batch_{int(time.time())}"
    
    # Store job info
    audio_jobs[job_id] = {
        'job_id': job_id,
        'status': 'queued',
        'input_paths': file_paths,
        'output_dir': output_dir,
        'file_count': len(file_paths),
        'created_at': time.time()
    }
    
    # Start batch processing in the background
    async def process_batch():
        try:
            # Update job status
            audio_jobs[job_id]['status'] = 'processing'
            
            # Initialize Audio Enhancer
            config = get_enhancer_config()
            enhancer = AudioEnhancer(config)
            
            # Process options
            options = {
                'apply_noise_reduction': request.apply_noise_reduction,
                'noise_reduction': request.noise_reduction.dict() if request.noise_reduction else {},
                'apply_voice_enhancement': request.apply_voice_enhancement,
                'voice_enhancement': request.voice_enhancement.dict() if request.voice_enhancement else {},
                'apply_dynamics_processing': request.apply_dynamics_processing,
                'dynamics_processing': request.dynamics_processing.dict() if request.dynamics_processing else {}
            }
            
            # Process the files in batch
            result = await enhancer.batch_process(
                file_paths=file_paths,
                options=options,
                output_dir=output_dir
            )
            
            # Update job with results
            audio_jobs[job_id].update({
                'status': 'completed',
                'successful': len(result['successful']),
                'failed': len(result['failed']),
                'results': result,
                'completed_at': time.time()
            })
            
        except Exception as e:
            logger.error(f"Error processing batch job {job_id}: {str(e)}")
            audio_jobs[job_id].update({
                'status': 'failed',
                'error': str(e),
                'completed_at': time.time()
            })
    
    # Start the batch processing task
    background_tasks.add_task(process_batch)
    
    return audio_jobs[job_id]

@router.post("/process-dynamics", response_class=JSONResponse)
async def process_dynamics(
    background_tasks: BackgroundTasks,
    request: DynamicsProcessRequest,
    file_path: str = Query(..., description="Path to the uploaded audio or video file"),
    current_user: User = Depends(get_current_user)
):
    """
    Apply dynamics processing to improve audio levels.
    
    Args:
        request: Dynamics processing parameters
        file_path: Path to the uploaded audio or video file
        
    Returns:
        Job status information
    """
    # Validate file path
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Create output directory
    output_dir = get_temp_path("dynamics_processed")
    
    # Create a unique job ID
    job_id = f"dynamics_{int(time.time())}_{os.path.basename(file_path)}"
    
    # Store job info
    audio_jobs[job_id] = {
        'job_id': job_id,
        'status': 'queued',
        'input_path': file_path,
        'output_dir': output_dir,
        'created_at': time.time()
    }
    
    # Start processing in the background
    async def process_dynamics_job():
        try:
            # Update job status
            audio_jobs[job_id]['status'] = 'processing'
            
            # Determine if it's a video or audio file
            is_video = os.path.splitext(file_path)[1].lower() in [
                '.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv'
            ]
            
            # Generate output path
            filename = os.path.basename(file_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_dynamics{ext}")
            
            # Initialize dynamics processor with parameters
            kwargs = {}
            for field in ['comp_threshold', 'comp_ratio', 'comp_attack', 'comp_release', 'preset']:
                value = getattr(request, field, None)
                if value is not None:
                    kwargs[field] = value
                    
            processor = DynamicsProcessor(**kwargs)
            
            # For video files, we need to extract audio, process it, and merge it back
            if is_video:
                # Extract audio
                audio_path = os.path.join(output_dir, f"{name}_audio.wav")
                
                ffmpeg_cmd = [
                    settings.ffmpeg_path,
                    "-i", file_path,
                    "-q:a", "0",
                    "-map", "a",
                    "-vn",
                    audio_path
                ]
                
                extract_process = await asyncio.create_subprocess_exec(
                    *ffmpeg_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await extract_process.communicate()
                
                if extract_process.returncode != 0:
                    logger.error(f"Failed to extract audio: {stderr.decode()}")
                    audio_jobs[job_id].update({
                        'status': 'failed',
                        'error': "Failed to extract audio from video",
                        'completed_at': time.time()
                    })
                    return
                
                # Process audio
                enhanced_audio = os.path.join(output_dir, f"{name}_processed_audio.wav")
                
                result = processor.process_audio(
                    audio_path=audio_path,
                    output_path=enhanced_audio,
                    apply_compression=request.apply_compression,
                    apply_limiting=request.apply_limiting,
                    apply_expansion=request.apply_expansion,
                    apply_gating=request.apply_gating,
                    target_loudness=request.target_loudness,
                    dry_wet_mix=request.dry_wet_mix
                )
                
                if result.get("status") != "success":
                    audio_jobs[job_id].update({
                        'status': 'failed',
                        'error': f"Failed to process audio: {result.get('error', 'Unknown error')}",
                        'completed_at': time.time()
                    })
                    return
                
                # Merge audio back with video
                merge_cmd = [
                    settings.ffmpeg_path,
                    "-i", file_path,
                    "-i", enhanced_audio,
                    "-c:v", "copy",  # Copy video stream without re-encoding
                    "-map", "0:v",   # Use video from first input
                    "-map", "1:a",   # Use audio from second input
                    "-y",            # Overwrite output if exists
                    output_path
                ]
                
                merge_process = await asyncio.create_subprocess_exec(
                    *merge_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await merge_process.communicate()
                
                if merge_process.returncode != 0:
                    logger.error(f"Failed to merge audio with video: {stderr.decode()}")
                    audio_jobs[job_id].update({
                        'status': 'failed',
                        'error': "Failed to merge processed audio with video",
                        'completed_at': time.time()
                    })
                    return
                
                # Update job results
                audio_jobs[job_id].update({
                    'status': 'completed',
                    'output_path': output_path,
                    'processing_details': result,
                    'completed_at': time.time()
                })
                
                # Clean up intermediate files
                for temp_file in [audio_path, enhanced_audio]:
                    if os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                        except Exception as e:
                            logger.warning(f"Failed to remove temp file {temp_file}: {str(e)}")
                
            else:
                # Process audio file directly
                result = processor.process_audio(
                    audio_path=file_path,
                    output_path=output_path,
                    apply_compression=request.apply_compression,
                    apply_limiting=request.apply_limiting,
                    apply_expansion=request.apply_expansion,
                    apply_gating=request.apply_gating,
                    target_loudness=request.target_loudness,
                    dry_wet_mix=request.dry_wet_mix
                )
                
                if result.get("status") != "success":
                    audio_jobs[job_id].update({
                        'status': 'failed',
                        'error': f"Failed to process audio: {result.get('error', 'Unknown error')}",
                        'completed_at': time.time()
                    })
                    return
                
                # Update job results
                audio_jobs[job_id].update({
                    'status': 'completed',
                    'output_path': output_path,
                    'processing_details': result,
                    'completed_at': time.time()
                })
            
        except Exception as e:
            logger.error(f"Error processing dynamics: {str(e)}")
            audio_jobs[job_id].update({
                'status': 'failed',
                'error': str(e),
                'completed_at': time.time()
            })
    
    # Start the background task
    background_tasks.add_task(process_dynamics_job)
    
    return audio_jobs[job_id]

@router.post("/analyze-dynamics", response_class=JSONResponse)
async def analyze_audio_dynamics(
    file_path: str = Query(..., description="Path to the audio or video file to analyze"),
    current_user: User = Depends(get_current_user)
):
    """
    Analyze audio dynamics to identify characteristics and recommend processing settings.
    
    Args:
        file_path: Path to the audio or video file
        
    Returns:
        Analysis results
    """
    # Validate file path
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine if it's a video or audio file
    is_video = os.path.splitext(file_path)[1].lower() in [
        '.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv'
    ]
    
    try:
        # For video files, extract audio first
        if is_video:
            # Create a temporary directory
            temp_dir = get_temp_path("dynamics_analysis")
            
            # Extract audio to a temporary file
            audio_path = os.path.join(temp_dir, f"{os.path.basename(file_path)}_audio.wav")
            
            # Run ffmpeg to extract audio
            ffmpeg_cmd = [
                settings.ffmpeg_path,
                "-i", file_path,
                "-q:a", "0",
                "-map", "a",
                "-vn",
                audio_path
            ]
            
            extract_process = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await extract_process.communicate()
            
            if extract_process.returncode != 0:
                logger.error(f"Failed to extract audio: {stderr.decode()}")
                raise HTTPException(status_code=500, detail="Failed to extract audio from video")
            
            # Initialize dynamics processor
            processor = DynamicsProcessor()
            
            # Analyze the extracted audio
            result = processor.analyze_dynamics(audio_path)
            
            # Clean up
            if os.path.exists(audio_path):
                os.remove(audio_path)
                
            # Clean up temp dir if empty
            try:
                os.rmdir(temp_dir)
            except:
                pass
        else:
            # Initialize dynamics processor
            processor = DynamicsProcessor()
            
            # Analyze audio file directly
            result = processor.analyze_dynamics(file_path)
        
        # Return analysis results
        if result.get("status") != "success":
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to analyze audio dynamics: {result.get('error', 'Unknown error')}"
            )
            
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing audio dynamics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing audio dynamics: {str(e)}")

@router.post("/classify-genre", response_class=JSONResponse)
async def classify_audio_genre(
    request: GenreClassificationRequest,
    file_path: str = Query(..., description="Path to the audio or video file to classify"),
    current_user: User = Depends(get_current_user)
):
    """
    Classify the genre of an audio file or the audio track of a video.
    
    This endpoint analyzes the audio and determines its genre characteristics,
    returning a list of potential genres and their probabilities.
    """
    # Validate file path
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        # Configure the genre classifier
        config = {
            'ffmpeg_path': settings.FFMPEG_PATH,
            'ffprobe_path': settings.FFPROBE_PATH,
            'temp_dir': settings.TEMP_DIR
        }
        
        classifier = GenreClassifier(config)
        
        # Run classification
        classification = classifier.classify_genre(file_path, top_n=request.top_n)
        
        # Check if classification was successful
        if classification['status'] != 'success':
            raise HTTPException(
                status_code=500, 
                detail=classification.get('error', 'Genre classification failed')
            )
        
        return JSONResponse(
            content={
                "status": "success",
                "file_path": file_path,
                "primary_genre": classification['primary_genre'],
                "top_genres": classification['top_genres'],
                "audio_features": classification.get('feature_summary', {})
            }
        )
        
    except Exception as e:
        logger.error(f"Genre classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Genre classification failed: {str(e)}")

@router.post("/recommend-genres", response_class=JSONResponse)
async def recommend_music_genres(
    request: GenreRecommendationRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Recommend music genres based on video content mood and genre.
    
    This endpoint helps content creators find appropriate music genres
    that match their video content's mood and genre.
    """
    try:
        # Configure the genre classifier
        config = {
            'ffmpeg_path': settings.FFMPEG_PATH,
            'ffprobe_path': settings.FFPROBE_PATH,
            'temp_dir': settings.TEMP_DIR
        }
        
        classifier = GenreClassifier(config)
        
        # Get recommendations
        recommendations = classifier.recommend_genres_for_video(
            request.video_mood,
            request.video_genre,
            top_n=request.top_n
        )
        
        # Check if we got any recommendations
        if not recommendations:
            return JSONResponse(
                content={
                    "status": "warning",
                    "message": f"No genre recommendations found for {request.video_mood} {request.video_genre} content",
                    "recommendations": []
                }
            )
        
        return JSONResponse(
            content={
                "status": "success",
                "video_mood": request.video_mood,
                "video_genre": request.video_genre,
                "recommendations": recommendations
            }
        )
        
    except Exception as e:
        logger.error(f"Genre recommendation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Genre recommendation failed: {str(e)}")

@router.post("/detect-bpm", response_class=JSONResponse)
async def detect_audio_bpm(
    request: BPMDetectionRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Detect the BPM (tempo) of an audio or video file.
    
    This endpoint analyzes the audio to determine its tempo characteristics,
    returning the BPM and tempo category.
    """
    file_path = request.file_path
    
    # Validate file path
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        # Configure the BPM detector
        config = {
            'ffmpeg_path': settings.FFMPEG_PATH,
            'ffprobe_path': settings.FFPROBE_PATH,
            'temp_dir': settings.TEMP_DIR
        }
        
        detector = BPMDetector(config)
        
        # Run BPM detection
        detection = detector.detect_bpm(file_path)
        
        # Check if detection was successful
        if detection['status'] != 'success':
            raise HTTPException(
                status_code=500, 
                detail=detection.get('error', 'BPM detection failed')
            )
        
        return JSONResponse(
            content={
                "status": "success",
                "file_path": file_path,
                "bpm": detection['bpm'],
                "category": detection['category'],
                "confidence": detection.get('confidence', None),
                "range": detection['range']
            }
        )
        
    except Exception as e:
        logger.error(f"BPM detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"BPM detection failed: {str(e)}")

@router.post("/match-bpm", response_class=JSONResponse)
async def match_audio_bpm(
    request: BPMMatchingRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Find audio files that match a target BPM (tempo).
    
    This endpoint analyzes multiple audio files and returns those that match
    the specified target BPM within the given tolerance range.
    """
    # Validate file paths
    for file_path in request.file_paths:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    
    try:
        # Configure the BPM detector
        config = {
            'ffmpeg_path': settings.FFMPEG_PATH,
            'ffprobe_path': settings.FFPROBE_PATH,
            'temp_dir': settings.TEMP_DIR
        }
        
        detector = BPMDetector(config)
        
        # Match tracks to target BPM
        match_result = detector.find_matching_tracks(
            target_bpm=request.target_bpm,
            audio_files=request.file_paths,
            tolerance=request.tolerance,
            match_style=request.match_style
        )
        
        return JSONResponse(content=match_result)
        
    except Exception as e:
        logger.error(f"BPM matching error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"BPM matching failed: {str(e)}")

@router.post("/suggest-bpm", response_class=JSONResponse)
async def suggest_bpm_for_content(
    request: BPMSuggestionRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Suggest appropriate BPM (tempo) ranges for different content types.
    
    This endpoint provides recommendations for appropriate music tempos
    based on the type of video content.
    """
    try:
        # Configure the BPM detector
        config = {
            'ffmpeg_path': settings.FFMPEG_PATH,
            'ffprobe_path': settings.FFPROBE_PATH,
            'temp_dir': settings.TEMP_DIR
        }
        
        detector = BPMDetector(config)
        
        # Get BPM suggestion
        suggestion = detector.suggest_bpm_for_content(request.content_type)
        
        return JSONResponse(content=suggestion)
        
    except Exception as e:
        logger.error(f"BPM suggestion error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"BPM suggestion failed: {str(e)}")

@router.post("/adjust-music-volume", response_class=JSONResponse)
async def adjust_music_volume_for_speech(
    request: VolumeAdjustmentRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Dynamically adjust music volume during speech segments in a video.
    
    This endpoint automatically lowers music volume when speech is detected and
    raises it during non-speech segments, ensuring dialogue intelligibility
    while maintaining background music.
    """
    # Validate file paths
    if not os.path.exists(request.video_path):
        raise HTTPException(status_code=404, detail=f"Video file not found: {request.video_path}")
    
    if not os.path.exists(request.music_path):
        raise HTTPException(status_code=404, detail=f"Music file not found: {request.music_path}")
    
    # Create output path
    output_filename = (
        f"speech_music_mix_{os.path.splitext(os.path.basename(request.video_path))[0]}_{int(time.time())}.mp4"
    )
    temp_dir = os.path.join(settings.TEMP_DIR, f"volume_adj_{int(time.time())}")
    os.makedirs(temp_dir, exist_ok=True)
    output_path = os.path.join(temp_dir, output_filename)
    
    try:
        # Configure the volume adjuster
        config = {
            'ffmpeg_path': settings.FFMPEG_PATH,
            'ffprobe_path': settings.FFPROBE_PATH,
            'temp_dir': temp_dir,
            'default_music_volume': request.default_volume,
            'ducking_amount': request.ducking_amount,
            'fade_in_time': request.fade_in_time,
            'fade_out_time': request.fade_out_time
        }
        
        # Create job ID
        job_id = f"volume_adj_{int(time.time())}"
        
        # Set up background processing for volume adjustment
        def process_volume_adjustment():
            try:
                volume_adjuster = VolumeAdjuster(config)
                
                # Process the video with dynamic volume adjustment
                result = volume_adjuster.adjust_music_for_speech(
                    video_path=request.video_path,
                    music_path=request.music_path,
                    output_path=output_path,
                    music_start_time=request.music_start_time,
                    music_end_time=request.music_end_time,
                    keep_original_audio=request.keep_original_audio
                )
                
                # Update job status
                audio_jobs[job_id] = {
                    "job_id": job_id,
                    "status": "completed" if result.get("status") == "success" else "failed",
                    "input_path": request.video_path,
                    "output_path": output_path if result.get("status") == "success" else None,
                    "created_at": time.time(),
                    "error": result.get("error") if result.get("status") != "success" else None,
                    "details": {
                        "speech_segments": result.get("speech_segments", []),
                        "volume_points": result.get("volume_points", [])
                    }
                }
                
            except Exception as e:
                logger.error(f"Volume adjustment error: {str(e)}")
                audio_jobs[job_id] = {
                    "job_id": job_id,
                    "status": "failed",
                    "input_path": request.video_path,
                    "output_path": None,
                    "created_at": time.time(),
                    "error": str(e)
                }
        
        # Start background processing
        background_tasks.add_task(process_volume_adjustment)
        
        # Initialize job status
        audio_jobs[job_id] = {
            "job_id": job_id,
            "status": "processing",
            "input_path": request.video_path,
            "output_path": None,
            "created_at": time.time()
        }
        
        return {
            "status": "success",
            "message": "Volume adjustment job started",
            "job_id": job_id
        }
        
    except Exception as e:
        logger.error(f"Error starting volume adjustment job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting volume adjustment job: {str(e)}")

class ContentMoodAnalysisRequest(BaseModel):
    """Request model for content mood analysis"""
    file_path: str = Field(..., description="Path to the video file to analyze")
    include_audio_analysis: bool = Field(True, description="Include audio-based mood analysis")
    include_visual_analysis: bool = Field(True, description="Include visual-based mood analysis") 
    include_transcript_analysis: bool = Field(True, description="Include transcript-based mood analysis")
    transcript_path: Optional[str] = Field(None, description="Path to transcript JSON file (optional)")
    segment_duration: int = Field(5, ge=1, le=30, description="Duration of segments for timeline analysis (seconds)")
    detailed_results: bool = Field(False, description="Include detailed mood timeline and scores")

@router.post("/analyze-mood", response_class=JSONResponse)
async def analyze_content_mood(
    request: ContentMoodAnalysisRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Analyze the emotional mood and tone of video content.
    
    This endpoint analyzes the audio, visual, and transcript elements of a video
    to determine its emotional characteristics, which can be used for selecting
    appropriate music.
    """
    try:
        # Validate file path
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
        
        # Get transcript data if provided
        transcript = None
        if request.transcript_path:
            if not os.path.exists(request.transcript_path):
                raise HTTPException(status_code=404, detail=f"Transcript file not found: {request.transcript_path}")
            
            try:
                with open(request.transcript_path, 'r') as f:
                    transcript = json.load(f)
            except Exception as e:
                logger.error(f"Error loading transcript: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Error loading transcript: {str(e)}")
        
        # Configure mood analyzer
        config = {
            'ffmpeg_path': settings.FFMPEG_PATH,
            'ffprobe_path': settings.FFPROBE_PATH,
            'temp_dir': settings.TEMP_DIR,
            'segment_duration': request.segment_duration
        }
        
        mood_analyzer = MoodAnalyzer(config)
        
        # Analyze mood
        mood_results = mood_analyzer.analyze_mood(
            video_path=request.file_path,
            transcript=transcript,
            include_audio_analysis=request.include_audio_analysis,
            include_visual_analysis=request.include_visual_analysis,
            include_transcript_analysis=request.include_transcript_analysis
        )
        
        # Build response
        response = {
            "status": "success",
            "file_path": request.file_path,
            "primary_mood": mood_results.get("primary_mood", "unknown"),
            "valence": mood_results.get("valence", 0),
            "arousal": mood_results.get("arousal", 0),
            "mood_scores": mood_results.get("mood_scores", {}),
            "recommended_music_moods": mood_results.get("recommended_music_moods", [])
        }
        
        # Include detailed results if requested
        if request.detailed_results:
            response["detailed_results"] = {
                "timeline": mood_results.get("timeline", []),
                "audio_analysis": mood_results.get("audio_analysis", {}),
                "visual_analysis": mood_results.get("visual_analysis", {}),
                "transcript_analysis": mood_results.get("transcript_analysis", {})
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Error analyzing content mood: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing content mood: {str(e)}")

class EmotionalArcMappingRequest(BaseModel):
    """Request model for emotional arc mapping"""
    file_path: str = Field(..., description="Path to the video file to analyze")
    transcript_path: Optional[str] = Field(None, description="Path to transcript JSON file (optional)")
    segment_duration: int = Field(5, ge=1, le=30, description="Duration of segments for timeline analysis (seconds)")
    detect_key_moments: bool = Field(True, description="Detect key emotional moments and transitions")
    smooth_arc: bool = Field(True, description="Apply smoothing to the emotional arc")
    use_existing_mood_analysis: bool = Field(False, description="Use existing mood analysis if available")

@router.post("/map-emotional-arc", response_class=JSONResponse)
async def map_emotional_arc(
    request: EmotionalArcMappingRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Map the emotional arc of video content over time.
    
    This endpoint analyzes how emotions evolve throughout a video, creating a timeline
    of emotional shifts and key moments that can be used for synchronized music selection.
    """
    try:
        # Validate file path
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
        
        # Get transcript data if provided
        transcript = None
        if request.transcript_path:
            if not os.path.exists(request.transcript_path):
                raise HTTPException(status_code=404, detail=f"Transcript file not found: {request.transcript_path}")
            
            try:
                with open(request.transcript_path, 'r') as f:
                    transcript = json.load(f)
            except Exception as e:
                logger.error(f"Error loading transcript: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Error loading transcript: {str(e)}")
        
        # Configure emotional arc mapper
        config = {
            'ffmpeg_path': settings.FFMPEG_PATH,
            'ffprobe_path': settings.FFPROBE_PATH,
            'temp_dir': settings.TEMP_DIR,
            'segment_duration': request.segment_duration
        }
        
        # Create arc mapper
        arc_mapper = EmotionalArcMapper(config)
        
        # Get existing mood analysis if requested
        mood_analysis = None
        if request.use_existing_mood_analysis:
            # Try to find existing mood analysis in the cache (if implemented)
            # This would need a caching system to be implemented
            pass
        
        # Map emotional arc
        arc_results = arc_mapper.map_emotional_arc(
            video_path=request.file_path,
            transcript=transcript,
            mood_analysis=mood_analysis,
            segment_duration=request.segment_duration,
            detect_key_moments=request.detect_key_moments,
            smooth_arc=request.smooth_arc
        )
        
        # If there was an error in the mapping
        if arc_results.get("status") == "error":
            raise HTTPException(status_code=500, detail=arc_results.get("error", "Error mapping emotional arc"))
        
        # Build response (structure will follow the EmotionalArcMapper output)
        response = {
            "status": "success",
            "file_path": request.file_path,
            "emotional_arc": arc_results.get("emotional_arc", []),
            "arc_pattern": arc_results.get("arc_pattern", "unknown"),
            "arc_confidence": arc_results.get("arc_confidence", 0),
            "pattern_description": arc_results.get("pattern_description", ""),
            "emotional_dynamics": arc_results.get("emotional_dynamics", {})
        }
        
        # Include key moments if detected
        if request.detect_key_moments and "key_moments" in arc_results:
            response["key_moments"] = arc_results["key_moments"]
        
        # Include music cues if generated
        if "music_cues" in arc_results:
            response["music_cues"] = arc_results["music_cues"]
        
        return response
        
    except Exception as e:
        logger.error(f"Error mapping emotional arc: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error mapping emotional arc: {str(e)}")

# Music Library API models
class AddTrackRequest(BaseModel):
    """Request model for adding a track to the music library"""
    file_path: str = Field(..., description="Path to the audio file")
    title: str = Field(..., description="Title of the track")
    artist: str = Field(..., description="Artist of the track")
    mood: Optional[str] = Field(None, description="Mood of the track")
    genre: Optional[str] = Field(None, description="Genre of the track")
    bpm: Optional[float] = Field(None, description="Tempo (BPM) of the track")
    tags: Optional[List[str]] = Field(None, description="Tags for the track")
    description: Optional[str] = Field(None, description="Description of the track")
    copyright_free: bool = Field(False, description="Whether the track is copyright-free")
    license: Optional[str] = Field(None, description="License information for the track")

class SearchTracksRequest(BaseModel):
    """Request model for searching tracks in the music library"""
    mood: Optional[str] = Field(None, description="Mood to search for")
    tempo: Optional[float] = Field(None, description="Tempo (BPM) to search for")
    genre: Optional[str] = Field(None, description="Genre to search for")
    duration: Optional[float] = Field(None, description="Target duration in seconds")
    keywords: Optional[List[str]] = Field(None, description="Keywords to search for")
    max_results: int = Field(10, description="Maximum number of results to return")
    copyright_free_only: bool = Field(False, description="Only return copyright-free tracks")
    collection_id: Optional[str] = Field(None, description="ID of collection to search in")

class CreateCollectionRequest(BaseModel):
    """Request model for creating a music collection"""
    name: str = Field(..., description="Name of the collection")
    description: Optional[str] = Field(None, description="Description of the collection")
    tags: Optional[List[str]] = Field(None, description="Tags for the collection")

class UpdateCollectionRequest(BaseModel):
    """Request model for updating a music collection"""
    collection_id: str = Field(..., description="ID of the collection to update")
    name: Optional[str] = Field(None, description="New name for the collection")
    description: Optional[str] = Field(None, description="New description for the collection")
    tags: Optional[List[str]] = Field(None, description="New tags for the collection")

class AddToCollectionRequest(BaseModel):
    """Request model for adding tracks to a collection"""
    collection_id: str = Field(..., description="ID of the collection")
    track_ids: List[str] = Field(..., description="IDs of tracks to add to the collection")

class RemoveFromCollectionRequest(BaseModel):
    """Request model for removing tracks from a collection"""
    collection_id: str = Field(..., description="ID of the collection")
    track_ids: List[str] = Field(..., description="IDs of tracks to remove from the collection")

class UpdateTrackRequest(BaseModel):
    """Request model for updating a track in the music library"""
    track_id: str = Field(..., description="ID of the track to update")
    title: Optional[str] = Field(None, description="New title for the track")
    artist: Optional[str] = Field(None, description="New artist for the track")
    mood: Optional[str] = Field(None, description="New mood for the track")
    genre: Optional[str] = Field(None, description="New genre for the track")
    bpm: Optional[float] = Field(None, description="New tempo (BPM) for the track")
    tags: Optional[List[str]] = Field(None, description="New tags for the track")
    description: Optional[str] = Field(None, description="New description for the track")
    copyright_free: Optional[bool] = Field(None, description="Whether the track is copyright-free")
    license: Optional[str] = Field(None, description="License information for the track")

# Audio Fingerprinting API models
class GenerateFingerprintRequest(BaseModel):
    """Request model for generating an audio fingerprint"""
    file_path: str = Field(..., description="Path to the audio file")

class AddToDatabaseRequest(BaseModel):
    """Request model for adding a fingerprint to the database"""
    file_path: str = Field(..., description="Path to the audio file")
    track_id: str = Field(..., description="ID of the track")
    title: str = Field(..., description="Title of the track")
    artist: str = Field(..., description="Artist of the track")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class IdentifyAudioRequest(BaseModel):
    """Request model for identifying audio using fingerprinting"""
    file_path: str = Field(..., description="Path to the audio file")
    top_n: int = Field(5, ge=1, le=20, description="Maximum number of matches to return")
    min_similarity: float = Field(0.8, ge=0.0, le=1.0, description="Minimum similarity threshold")

class RemoveFromDatabaseRequest(BaseModel):
    """Request model for removing a fingerprint from the database"""
    track_id: str = Field(..., description="ID of the track to remove")

class CompareFingerprintsRequest(BaseModel):
    """Request model for comparing two audio fingerprints"""
    file_path1: str = Field(..., description="Path to the first audio file")
    file_path2: str = Field(..., description="Path to the second audio file")
    detailed_results: bool = Field(False, description="Include detailed comparison metrics")

# Music Library API endpoints
@router.post("/library/add-track", response_class=JSONResponse)
async def add_track_to_library(
    request: AddTrackRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Add a track to the music library.
    
    This endpoint adds a music track to the library with metadata
    such as title, artist, mood, genre, and BPM.
    """
    try:
        # Validate file path
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
        
        # Configure music library
        config = {
            'music_library_path': settings.MUSIC_LIBRARY_PATH,
            'ffmpeg_path': settings.FFMPEG_PATH,
            'ffprobe_path': settings.FFPROBE_PATH
        }
        
        # Create music library
        music_library = MusicLibrary(config)
        
        # Add track to library
        result = music_library.add_track(
            file_path=request.file_path,
            title=request.title,
            artist=request.artist,
            mood=request.mood,
            genre=request.genre,
            bpm=request.bpm,
            tags=request.tags,
            description=request.description
        )
        
        # Update track with additional metadata
        if result["status"] == "success" and (request.copyright_free or request.license):
            track_id = result["track_id"]
            track = music_library.get_track(track_id)
            if track:
                # Update with additional metadata
                track["copyright_free"] = request.copyright_free
                if request.license:
                    track["license"] = request.license
                # Save metadata
                music_library._save_metadata()
        
        return result
        
    except Exception as e:
        logger.error(f"Error adding track to library: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding track to library: {str(e)}")

@router.post("/library/search", response_class=JSONResponse)
async def search_music_library(
    request: SearchTracksRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Search for tracks in the music library.
    
    This endpoint searches for music tracks in the library based on
    criteria such as mood, tempo, genre, duration, and keywords.
    """
    try:
        # Configure music library
        config = {
            'music_library_path': settings.MUSIC_LIBRARY_PATH,
            'ffmpeg_path': settings.FFMPEG_PATH,
            'ffprobe_path': settings.FFPROBE_PATH
        }
        
        # Create music library
        music_library = MusicLibrary(config)
        
        # Search for tracks
        results = music_library.search_tracks(
            mood=request.mood,
            tempo=request.tempo,
            genre=request.genre,
            duration=request.duration,
            keywords=request.keywords,
            max_results=request.max_results
        )
        
        # Filter by copyright status if requested
        if request.copyright_free_only and results["status"] == "success":
            results["tracks"] = [t for t in results["tracks"] if t.get("copyright_free", False)]
            results["total_matches"] = len(results["tracks"])
        
        # Filter by collection if requested
        if request.collection_id and results["status"] == "success":
            collection = music_library.get_collection(request.collection_id)
            if not collection:
                raise HTTPException(status_code=404, detail=f"Collection not found: {request.collection_id}")
            
            collection_track_ids = collection.get("track_ids", [])
            results["tracks"] = [t for t in results["tracks"] if t.get("id") in collection_track_ids]
            results["total_matches"] = len(results["tracks"])
        
        return results
        
    except Exception as e:
        logger.error(f"Error searching music library: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching music library: {str(e)}")

@router.delete("/library/track/{track_id}", response_class=JSONResponse)
async def remove_track_from_library(
    track_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Remove a track from the music library.
    
    This endpoint removes a music track from the library by ID.
    """
    try:
        # Configure music library
        config = {
            'music_library_path': settings.MUSIC_LIBRARY_PATH,
            'ffmpeg_path': settings.FFMPEG_PATH,
            'ffprobe_path': settings.FFPROBE_PATH
        }
        
        # Create music library
        music_library = MusicLibrary(config)
        
        # Remove track from library
        result = music_library.remove_track(track_id)
        
        return result
        
    except Exception as e:
        logger.error(f"Error removing track from library: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error removing track from library: {str(e)}")

@router.patch("/library/track", response_class=JSONResponse)
async def update_track_metadata(
    request: UpdateTrackRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Update track metadata in the music library.
    
    This endpoint updates the metadata of a music track in the library.
    """
    try:
        # Configure music library
        config = {
            'music_library_path': settings.MUSIC_LIBRARY_PATH,
            'ffmpeg_path': settings.FFMPEG_PATH,
            'ffprobe_path': settings.FFPROBE_PATH
        }
        
        # Create music library
        music_library = MusicLibrary(config)
        
        # Get track
        track = music_library.get_track(request.track_id)
        if not track:
            raise HTTPException(status_code=404, detail=f"Track not found: {request.track_id}")
        
        # Update track metadata
        updated = False
        if request.title is not None:
            track["title"] = request.title
            updated = True
        if request.artist is not None:
            track["artist"] = request.artist
            updated = True
        if request.mood is not None:
            track["mood"] = request.mood
            updated = True
        if request.genre is not None:
            track["genre"] = request.genre
            updated = True
        if request.bpm is not None:
            track["bpm"] = request.bpm
            updated = True
        if request.tags is not None:
            track["tags"] = request.tags
            updated = True
        if request.description is not None:
            track["description"] = request.description
            updated = True
        if request.copyright_free is not None:
            track["copyright_free"] = request.copyright_free
            updated = True
        if request.license is not None:
            track["license"] = request.license
            updated = True
        
        # Save metadata if updated
        if updated:
            music_library._save_metadata()
        
        return {
            "status": "success",
            "message": f"Track '{track.get('title')}' updated",
            "track_id": request.track_id
        }
        
    except Exception as e:
        logger.error(f"Error updating track metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating track metadata: {str(e)}")

@router.post("/library/collection/create", response_class=JSONResponse)
async def create_music_collection(
    request: CreateCollectionRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Create a music collection in the library.
    
    This endpoint creates a new collection for organizing music tracks.
    """
    try:
        # Configure music library
        config = {
            'music_library_path': settings.MUSIC_LIBRARY_PATH,
            'ffmpeg_path': settings.FFMPEG_PATH,
            'ffprobe_path': settings.FFPROBE_PATH
        }
        
        # Create music library
        music_library = MusicLibrary(config)
        
        # Create collection
        result = music_library.create_collection(
            name=request.name,
            description=request.description,
            tags=request.tags
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error creating music collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating music collection: {str(e)}")

@router.patch("/library/collection", response_class=JSONResponse)
async def update_music_collection(
    request: UpdateCollectionRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Update a music collection in the library.
    
    This endpoint updates the metadata of a music collection.
    """
    try:
        # Configure music library
        config = {
            'music_library_path': settings.MUSIC_LIBRARY_PATH,
            'ffmpeg_path': settings.FFMPEG_PATH,
            'ffprobe_path': settings.FFPROBE_PATH
        }
        
        # Create music library
        music_library = MusicLibrary(config)
        
        # Update collection
        result = music_library.update_collection(
            collection_id=request.collection_id,
            name=request.name,
            description=request.description,
            tags=request.tags
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error updating music collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating music collection: {str(e)}")

@router.delete("/library/collection/{collection_id}", response_class=JSONResponse)
async def delete_music_collection(
    collection_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Delete a music collection from the library.
    
    This endpoint deletes a music collection by ID.
    """
    try:
        # Configure music library
        config = {
            'music_library_path': settings.MUSIC_LIBRARY_PATH,
            'ffmpeg_path': settings.FFMPEG_PATH,
            'ffprobe_path': settings.FFPROBE_PATH
        }
        
        # Create music library
        music_library = MusicLibrary(config)
        
        # Delete collection
        result = music_library.delete_collection(collection_id)
        
        return result
        
    except Exception as e:
        logger.error(f"Error deleting music collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting music collection: {str(e)}")

@router.post("/library/collection/add-tracks", response_class=JSONResponse)
async def add_tracks_to_collection(
    request: AddToCollectionRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Add tracks to a music collection.
    
    This endpoint adds tracks to a music collection by ID.
    """
    try:
        # Configure music library
        config = {
            'music_library_path': settings.MUSIC_LIBRARY_PATH,
            'ffmpeg_path': settings.FFMPEG_PATH,
            'ffprobe_path': settings.FFPROBE_PATH
        }
        
        # Create music library
        music_library = MusicLibrary(config)
        
        # Add tracks to collection
        result = music_library.add_tracks_to_collection(
            collection_id=request.collection_id,
            track_ids=request.track_ids
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error adding tracks to collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding tracks to collection: {str(e)}")

@router.post("/library/collection/remove-tracks", response_class=JSONResponse)
async def remove_tracks_from_collection(
    request: RemoveFromCollectionRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Remove tracks from a music collection.
    
    This endpoint removes tracks from a music collection by ID.
    """
    try:
        # Configure music library
        config = {
            'music_library_path': settings.MUSIC_LIBRARY_PATH,
            'ffmpeg_path': settings.FFMPEG_PATH,
            'ffprobe_path': settings.FFPROBE_PATH
        }
        
        # Create music library
        music_library = MusicLibrary(config)
        
        # Remove tracks from collection
        result = music_library.remove_tracks_from_collection(
            collection_id=request.collection_id,
            track_ids=request.track_ids
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error removing tracks from collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error removing tracks from collection: {str(e)}")

@router.get("/library/collections", response_class=JSONResponse)
async def get_all_collections(
    current_user: User = Depends(get_current_user)
):
    """
    Get all music collections in the library.
    
    This endpoint returns all music collections and their metadata.
    """
    try:
        # Configure music library
        config = {
            'music_library_path': settings.MUSIC_LIBRARY_PATH,
            'ffmpeg_path': settings.FFMPEG_PATH,
            'ffprobe_path': settings.FFPROBE_PATH
        }
        
        # Create music library
        music_library = MusicLibrary(config)
        
        # Get all collections
        collections = music_library.get_all_collections()
        
        return {
            "status": "success",
            "collections": collections
        }
        
    except Exception as e:
        logger.error(f"Error getting music collections: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting music collections: {str(e)}")

@router.get("/library/collection/{collection_id}", response_class=JSONResponse)
async def get_collection(
    collection_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get a music collection by ID.
    
    This endpoint returns a music collection with its tracks.
    """
    try:
        # Configure music library
        config = {
            'music_library_path': settings.MUSIC_LIBRARY_PATH,
            'ffmpeg_path': settings.FFMPEG_PATH,
            'ffprobe_path': settings.FFPROBE_PATH
        }
        
        # Create music library
        music_library = MusicLibrary(config)
        
        # Get collection
        collection = music_library.get_collection(collection_id)
        if not collection:
            raise HTTPException(status_code=404, detail=f"Collection not found: {collection_id}")
        
        # Get tracks in collection
        tracks = []
        for track_id in collection.get("track_ids", []):
            track = music_library.get_track(track_id)
            if track:
                tracks.append(track)
        
        collection["tracks"] = tracks
        
        return {
            "status": "success",
            "collection": collection
        }
        
    except Exception as e:
        logger.error(f"Error getting music collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting music collection: {str(e)}")

@router.get("/library/tracks", response_class=JSONResponse)
async def get_all_tracks(
    current_user: User = Depends(get_current_user)
):
    """
    Get all tracks in the music library.
    
    This endpoint returns metadata for all tracks in the library.
    """
    try:
        # Configure music library
        config = {
            'music_library_path': settings.MUSIC_LIBRARY_PATH,
            'ffmpeg_path': settings.FFMPEG_PATH,
            'ffprobe_path': settings.FFPROBE_PATH
        }
        
        # Create music library
        music_library = MusicLibrary(config)
        
        return {
            "status": "success",
            "tracks": music_library.tracks
        }
        
    except Exception as e:
        logger.error(f"Error getting music tracks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting music tracks: {str(e)}")

@router.get("/library/track/{track_id}", response_class=JSONResponse)
async def get_track(
    track_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get a track by ID.
    
    This endpoint returns metadata for a specific track.
    """
    try:
        # Configure music library
        config = {
            'music_library_path': settings.MUSIC_LIBRARY_PATH,
            'ffmpeg_path': settings.FFMPEG_PATH,
            'ffprobe_path': settings.FFPROBE_PATH
        }
        
        # Create music library
        music_library = MusicLibrary(config)
        
        # Get track
        track = music_library.get_track(track_id)
        if not track:
            raise HTTPException(status_code=404, detail=f"Track not found: {track_id}")
        
        return {
            "status": "success",
            "track": track
        }
        
    except Exception as e:
        logger.error(f"Error getting track: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting track: {str(e)}")

# Audio Fingerprinting API endpoints
@router.post("/fingerprint/generate", response_class=JSONResponse)
async def generate_audio_fingerprint(
    request: GenerateFingerprintRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Generate an audio fingerprint.
    
    This endpoint generates a unique fingerprint for an audio file,
    which can be used for identification and comparison.
    """
    try:
        # Validate file path
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
        
        # Configure audio fingerprinter
        config = {
            'fingerprint_db_path': os.path.join(settings.DATA_DIR, 'fingerprint_db'),
            'ffmpeg_path': settings.FFMPEG_PATH,
            'ffprobe_path': settings.FFPROBE_PATH
        }
        
        # Create audio fingerprinter
        fingerprinter = AudioFingerprinter(config)
        
        # Generate fingerprint
        result = fingerprinter.generate_fingerprint(request.file_path)
        
        # If the fingerprint includes large vectors, summarize them for the API response
        if result["status"] in ["success", "partial_success"] and "fingerprint" in result:
            # Preserve the hash but summarize feature data to keep the response size manageable
            fingerprint_data = result["fingerprint"]
            if "vector" in fingerprint_data:
                # Only include vector length and a few sample values
                vector = fingerprint_data["vector"]
                fingerprint_data["vector_info"] = {
                    "length": len(vector),
                    "sample": vector[:5] if len(vector) >= 5 else vector
                }
                del fingerprint_data["vector"]
            
            # Summarize feature data if present
            if "features" in fingerprint_data:
                for feature_name, feature_vector in fingerprint_data["features"].items():
                    fingerprint_data["features"][feature_name] = {
                        "length": len(feature_vector),
                        "mean": np.mean(feature_vector) if len(feature_vector) > 0 else 0,
                        "min": min(feature_vector) if len(feature_vector) > 0 else 0,
                        "max": max(feature_vector) if len(feature_vector) > 0 else 0
                    }
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating audio fingerprint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating audio fingerprint: {str(e)}")

@router.post("/fingerprint/add-to-database", response_class=JSONResponse)
async def add_fingerprint_to_database(
    request: AddToDatabaseRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Add an audio fingerprint to the database.
    
    This endpoint adds an audio fingerprint to the database for later identification.
    """
    try:
        # Validate file path
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
        
        # Configure audio fingerprinter
        config = {
            'fingerprint_db_path': os.path.join(settings.DATA_DIR, 'fingerprint_db'),
            'ffmpeg_path': settings.FFMPEG_PATH,
            'ffprobe_path': settings.FFPROBE_PATH
        }
        
        # Create audio fingerprinter
        fingerprinter = AudioFingerprinter(config)
        
        # Add to database
        result = fingerprinter.add_to_database(
            audio_path=request.file_path,
            track_id=request.track_id,
            title=request.title,
            artist=request.artist,
            metadata=request.metadata
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error adding fingerprint to database: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding fingerprint to database: {str(e)}")

@router.post("/fingerprint/identify", response_class=JSONResponse)
async def identify_audio_with_fingerprint(
    request: IdentifyAudioRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Identify an audio file using fingerprinting.
    
    This endpoint compares the fingerprint of an audio file to the database
    to identify potential matches.
    """
    try:
        # Validate file path
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
        
        # Configure audio fingerprinter
        config = {
            'fingerprint_db_path': os.path.join(settings.DATA_DIR, 'fingerprint_db'),
            'ffmpeg_path': settings.FFMPEG_PATH,
            'ffprobe_path': settings.FFPROBE_PATH,
            'similarity_threshold': request.min_similarity
        }
        
        # Create audio fingerprinter
        fingerprinter = AudioFingerprinter(config)
        
        # Identify audio
        result = fingerprinter.identify_audio(request.file_path)
        
        # Limit matches based on top_n
        if result["status"] == "success" and "matches" in result:
            result["matches"] = result["matches"][:request.top_n]
        
        return result
        
    except Exception as e:
        logger.error(f"Error identifying audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error identifying audio: {str(e)}")

@router.post("/fingerprint/remove-from-database", response_class=JSONResponse)
async def remove_fingerprint_from_database(
    request: RemoveFromDatabaseRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Remove an audio fingerprint from the database.
    
    This endpoint removes a fingerprint from the database by track ID.
    """
    try:
        # Configure audio fingerprinter
        config = {
            'fingerprint_db_path': os.path.join(settings.DATA_DIR, 'fingerprint_db'),
            'ffmpeg_path': settings.FFMPEG_PATH,
            'ffprobe_path': settings.FFPROBE_PATH
        }
        
        # Create audio fingerprinter
        fingerprinter = AudioFingerprinter(config)
        
        # Remove from database
        result = fingerprinter.remove_from_database(request.track_id)
        
        return result
        
    except Exception as e:
        logger.error(f"Error removing fingerprint from database: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error removing fingerprint from database: {str(e)}")

@router.post("/fingerprint/compare", response_class=JSONResponse)
async def compare_audio_fingerprints(
    request: CompareFingerprintsRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Compare two audio files using fingerprinting.
    
    This endpoint compares the fingerprints of two audio files to determine similarity.
    """
    try:
        # Validate file paths
        if not os.path.exists(request.file_path1):
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path1}")
        if not os.path.exists(request.file_path2):
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path2}")
        
        # Configure audio fingerprinter
        config = {
            'fingerprint_db_path': os.path.join(settings.DATA_DIR, 'fingerprint_db'),
            'ffmpeg_path': settings.FFMPEG_PATH,
            'ffprobe_path': settings.FFPROBE_PATH
        }
        
        # Create audio fingerprinter
        fingerprinter = AudioFingerprinter(config)
        
        # Generate fingerprints for both files
        fingerprint1 = fingerprinter.generate_fingerprint(request.file_path1)
        fingerprint2 = fingerprinter.generate_fingerprint(request.file_path2)
        
        # Check if fingerprints were successfully generated
        if fingerprint1["status"] not in ["success", "partial_success"]:
            return {
                "status": "error",
                "error": f"Failed to generate fingerprint for {request.file_path1}"
            }
        if fingerprint2["status"] not in ["success", "partial_success"]:
            return {
                "status": "error",
                "error": f"Failed to generate fingerprint for {request.file_path2}"
            }
        
        # Compare fingerprints
        comparison = fingerprinter.compare_fingerprints(
            fingerprint1.get("fingerprint", {}),
            fingerprint2.get("fingerprint", {})
        )
        
        # Include file paths in the result
        comparison["file_path1"] = request.file_path1
        comparison["file_path2"] = request.file_path2
        
        # Include detailed results if requested
        if not request.detailed_results:
            # Keep only essential information
            essential_keys = ["status", "file_path1", "file_path2", "similarity", "distance", "is_match"]
            comparison = {key: comparison[key] for key in essential_keys if key in comparison}
        
        return comparison
        
    except Exception as e:
        logger.error(f"Error comparing audio fingerprints: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error comparing audio fingerprints: {str(e)}")

@router.get("/fingerprint/database-info", response_class=JSONResponse)
async def get_fingerprint_database_info(
    current_user: User = Depends(get_current_user)
):
    """
    Get information about the fingerprint database.
    
    This endpoint returns information about the fingerprint database,
    including the number of fingerprints and basic statistics.
    """
    try:
        # Configure audio fingerprinter
        config = {
            'fingerprint_db_path': os.path.join(settings.DATA_DIR, 'fingerprint_db'),
            'ffmpeg_path': settings.FFMPEG_PATH,
            'ffprobe_path': settings.FFPROBE_PATH
        }
        
        # Create audio fingerprinter
        fingerprinter = AudioFingerprinter(config)
        
        # Load fingerprint database
        database = fingerprinter._load_fingerprint_database()
        
        # Compute database statistics
        database_info = {
            "status": "success",
            "count": len(database),
            "database_path": fingerprinter.fingerprint_db_path,
            "track_ids": [item.get("track_id") for item in database],
            "titles": [item.get("title") for item in database],
            "artists": [item.get("artist") for item in database],
            "updated_at": max([item.get("timestamp", "1970-01-01T00:00:00") for item in database]) if database else None
        }
        
        return database_info
        
    except Exception as e:
        logger.error(f"Error getting fingerprint database info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting fingerprint database info: {str(e)}") 

# External Music Service API endpoints
class ExternalMusicSearchRequest(BaseModel):
    query: str
    service: Optional[str] = None
    mood: Optional[str] = None
    genre: Optional[str] = None
    bpm: Optional[int] = None
    duration: Optional[int] = None
    license_type: Optional[str] = Field(None, description="Filter by license type (e.g., 'cc', 'commercial')")
    max_results: int = Field(20, ge=1, le=100)
    use_cache: bool = True

class ExternalMusicDownloadRequest(BaseModel):
    track_id: str
    service: str
    output_filename: Optional[str] = None
    include_metadata: bool = True

class ExternalMusicImportRequest(BaseModel):
    track_id: str
    service: str
    collection_id: Optional[str] = None

class ExternalServiceInfoRequest(BaseModel):
    service: Optional[str] = None

@router.post("/external/search", response_class=JSONResponse)
def search_external_music(
    request: ExternalMusicSearchRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Search for music tracks across external music services.
    
    This endpoint allows searching for music tracks across multiple external 
    music services like Jamendo, Free Music Archive, and others. Results can
    be filtered by service, mood, genre, BPM, duration, and license type.
    
    The response includes track metadata and preview URLs.
    """
    try:
        # Initialize external music service
        from app.services.music.external_music_service import ExternalMusicService
        external_service = ExternalMusicService()
        
        # Search for tracks
        results = external_service.search_tracks(
            query=request.query,
            service=request.service,
            mood=request.mood,
            genre=request.genre,
            bpm=request.bpm,
            duration=request.duration,
            license_type=request.license_type,
            max_results=request.max_results,
            use_cache=request.use_cache
        )
        
        return results
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@router.post("/external/download", response_class=JSONResponse)
def download_external_music(
    request: ExternalMusicDownloadRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Download a music track from an external service.
    
    This endpoint downloads a track from an external music service to the server.
    The track is saved to the server's download directory and can optionally
    include metadata in a sidecar JSON file.
    
    The response includes the saved file path and size.
    """
    try:
        # Initialize external music service
        from app.services.music.external_music_service import ExternalMusicService
        external_service = ExternalMusicService()
        
        # Download track
        results = external_service.download_track(
            track_id=request.track_id,
            service=request.service,
            output_path=request.output_filename,
            metadata={"include_metadata": request.include_metadata} if request.include_metadata else None
        )
        
        return results
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@router.post("/external/import", response_class=JSONResponse)
def import_external_music(
    request: ExternalMusicImportRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Import a track from an external service to the local music library.
    
    This endpoint downloads a track from an external music service and adds
    it to the local music library. The track can optionally be added to a
    specific collection.
    
    The response includes the imported track information.
    """
    try:
        # Initialize external music service
        from app.services.music.external_music_service import ExternalMusicService
        external_service = ExternalMusicService()
        
        # Import track
        results = external_service.import_to_library(
            track_id=request.track_id,
            service=request.service,
            collection_id=request.collection_id
        )
        
        return results
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@router.post("/external/service-info", response_class=JSONResponse)
def get_external_service_info(
    request: ExternalServiceInfoRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Get information about supported external music services.
    
    This endpoint returns information about supported external music services,
    including service descriptions, API configurations, and whether API keys
    are configured.
    
    If a specific service is specified, only information for that service is returned.
    """
    try:
        # Initialize external music service
        from app.services.music.external_music_service import ExternalMusicService
        external_service = ExternalMusicService()
        
        # Get service info
        results = external_service.get_service_info(request.service)
        
        return results
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

# Music Recommendation Engine API endpoints
class VideoRecommendationRequest(BaseModel):
    """Request model for video-based music recommendations"""
    file_path: str = Field(..., description="Path to the video file")
    transcript_path: Optional[str] = Field(None, description="Path to transcript JSON file (optional)")
    user_id: Optional[str] = Field(None, description="User ID for personalized recommendations")
    target_duration: Optional[float] = Field(None, description="Target duration for the music")
    mood_override: Optional[str] = Field(None, description="Optional mood override")
    genre_override: Optional[str] = Field(None, description="Optional genre override")
    tempo_override: Optional[float] = Field(None, description="Optional tempo override")
    max_results: int = Field(10, ge=1, le=30, description="Maximum number of recommendations")
    copyright_free_only: bool = Field(False, description="Only return copyright-free tracks")
    collection_id: Optional[str] = Field(None, description="ID of collection to search in")
    use_emotional_arc: bool = Field(True, description="Whether to analyze emotional arc for better music selection")
    include_external_services: bool = Field(True, description="Whether to include results from external music services")
    segment_duration: Optional[int] = Field(5, ge=1, le=30, description="Duration of segments for timeline analysis (seconds)")
    diversity_level: float = Field(0.3, ge=0.0, le=1.0, description="Level of diversity in recommendations (0-1)")

class SimilarTrackRecommendationRequest(BaseModel):
    """Request model for similar track recommendations"""
    track_id: str = Field(..., description="ID of the reference track")
    source: str = Field("library", description="Source of the track ('library' or external service name)")
    user_id: Optional[str] = Field(None, description="User ID for personalized recommendations")
    max_results: int = Field(10, ge=1, le=30, description="Maximum number of recommendations")
    include_external_services: bool = Field(True, description="Whether to include results from external music services")

class RecommendationFeedbackRequest(BaseModel):
    """Request model for submitting feedback on recommendations"""
    user_id: str = Field(..., description="ID of the user submitting feedback")
    track_id: str = Field(..., description="ID of the track being rated")
    source: str = Field("library", description="Source of the track ('library' or external service name)")
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating (1-5)")
    liked: Optional[bool] = Field(None, description="Whether the user liked the track")
    used_in_project: Optional[bool] = Field(None, description="Whether the track was used in a project")
    context: Optional[Dict[str, Any]] = Field(None, description="Optional context information")

class UserPreferencesRequest(BaseModel):
    """Request model for getting user preferences"""
    user_id: str = Field(..., description="ID of the user")

class UserHistoryRequest(BaseModel):
    """Request model for getting user recommendation history"""
    user_id: str = Field(..., description="ID of the user")
    limit: int = Field(10, ge=1, le=50, description="Maximum number of history entries to return")

@router.post("/recommend/for-video", response_class=JSONResponse)
async def recommend_music_for_video(
    request: VideoRecommendationRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Recommend music tracks for a specific video.
    
    This endpoint analyzes the video content and provides personalized music
    recommendations based on content analysis and user preferences. It combines
    different recommendation sources including content-based filtering, user
    preferences, and external music services.
    
    The response includes a ranked list of recommended tracks and optional
    emotional timeline for dynamic soundtracks.
    """
    try:
        # Check if file exists
        if not os.path.exists(request.file_path):
            return JSONResponse(
                status_code=404,
                content={"status": "error", "message": f"File not found: {request.file_path}"}
            )
        
        # Load transcript if provided
        transcript = None
        if request.transcript_path:
            if os.path.exists(request.transcript_path):
                with open(request.transcript_path, 'r') as f:
                    transcript = json.load(f)
            else:
                return JSONResponse(
                    status_code=404,
                    content={"status": "error", "message": f"Transcript file not found: {request.transcript_path}"}
                )
        
        # Initialize music recommender
        from app.services.music.music_recommender import MusicRecommender
        recommender = MusicRecommender()
        
        # Get recommendations
        results = recommender.recommend_for_video(
            video_path=request.file_path,
            user_id=request.user_id,
            transcript=transcript,
            target_duration=request.target_duration,
            mood_override=request.mood_override,
            tempo_override=request.tempo_override,
            genre_override=request.genre_override,
            max_results=request.max_results,
            copyright_free_only=request.copyright_free_only,
            collection_id=request.collection_id,
            use_emotional_arc=request.use_emotional_arc,
            include_external_services=request.include_external_services,
            segment_duration=request.segment_duration,
            diversity_level=request.diversity_level
        )
        
        return results
    except Exception as e:
        logger.error(f"Error recommending music for video: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@router.post("/recommend/similar-to-track", response_class=JSONResponse)
async def recommend_similar_to_track(
    request: SimilarTrackRecommendationRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Recommend tracks similar to a specific track.
    
    This endpoint provides music recommendations based on similarity to a
    reference track. It uses attributes like mood, genre, and tempo to find
    similar tracks in the local library and optionally in external services.
    
    The response includes a ranked list of similar tracks with relevance scores.
    """
    try:
        # Initialize music recommender
        from app.services.music.music_recommender import MusicRecommender
        recommender = MusicRecommender()
        
        # Get recommendations
        results = recommender.recommend_similar_to_track(
            track_id=request.track_id,
            user_id=request.user_id,
            source=request.source,
            max_results=request.max_results,
            include_external_services=request.include_external_services
        )
        
        return results
    except Exception as e:
        logger.error(f"Error recommending similar tracks: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@router.post("/recommend/submit-feedback", response_class=JSONResponse)
async def submit_recommendation_feedback(
    request: RecommendationFeedbackRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Submit feedback on music recommendations.
    
    This endpoint allows users to provide feedback on recommended tracks,
    which helps improve future recommendations. Feedback can include ratings,
    likes/dislikes, and whether the track was used in a project.
    
    The system updates user preferences based on the feedback to provide
    more personalized recommendations in the future.
    """
    try:
        # Initialize music recommender
        from app.services.music.music_recommender import MusicRecommender
        recommender = MusicRecommender()
        
        # Submit feedback
        results = recommender.submit_feedback(
            user_id=request.user_id,
            track_id=request.track_id,
            source=request.source,
            rating=request.rating,
            liked=request.liked,
            used_in_project=request.used_in_project,
            context=request.context
        )
        
        return results
    except Exception as e:
        logger.error(f"Error submitting recommendation feedback: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@router.post("/recommend/user-preferences", response_class=JSONResponse)
async def get_user_preferences(
    request: UserPreferencesRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Get music preferences for a user.
    
    This endpoint returns the current music preferences for a specific user,
    including preferred moods, genres, artists, and favorite tracks.
    
    These preferences are used to provide personalized music recommendations.
    """
    try:
        # Initialize music recommender
        from app.services.music.music_recommender import MusicRecommender
        recommender = MusicRecommender()
        
        # Get user preferences
        results = recommender.get_user_preferences(request.user_id)
        
        return results
    except Exception as e:
        logger.error(f"Error getting user preferences: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@router.post("/recommend/user-history", response_class=JSONResponse)
async def get_user_recommendation_history(
    request: UserHistoryRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Get recommendation history for a user.
    
    This endpoint returns the history of music recommendations for a specific user,
    including the videos they were recommended for, the tracks that were recommended,
    and when the recommendations were made.
    
    This history can be used to avoid recommending the same tracks repeatedly and
    to improve future recommendations.
    """
    try:
        # Initialize music recommender
        from app.services.music.music_recommender import MusicRecommender
        recommender = MusicRecommender()
        
        # Get user recommendation history
        results = recommender.get_user_recommendation_history(
            user_id=request.user_id,
            limit=request.limit
        )
        
        return results
    except Exception as e:
        logger.error(f"Error getting user recommendation history: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

# Sound Effects Request Models
class AddSoundEffectRequest(BaseModel):
    file_path: str = Field(..., description="Path to the sound effect file")
    name: str = Field(..., description="Name of the sound effect")
    category: str = Field(..., description="Category of the sound effect")
    tags: Optional[List[str]] = Field(None, description="Tags associated with the sound effect")
    description: Optional[str] = Field(None, description="Description of the sound effect")
    duration: Optional[float] = Field(None, description="Duration of the sound effect in seconds")
    trigger_words: Optional[List[str]] = Field(None, description="Words that can trigger this sound effect")
    custom_library_id: Optional[str] = Field(None, description="ID of custom library")

class SearchSoundEffectsRequest(BaseModel):
    query: Optional[str] = Field(None, description="Text search query")
    category: Optional[str] = Field(None, description="Category to filter by")
    tags: Optional[List[str]] = Field(None, description="List of tags to filter by")
    min_duration: Optional[float] = Field(None, description="Minimum duration in seconds")
    max_duration: Optional[float] = Field(None, description="Maximum duration in seconds")
    max_results: int = Field(20, description="Maximum number of results to return")
    collection_id: Optional[str] = Field(None, description="ID of collection to search within")

class FindByTriggerWordsRequest(BaseModel):
    text: str = Field(..., description="Text to analyze for trigger words")
    max_results_per_word: int = Field(3, description="Maximum number of results per trigger word")
    case_sensitive: bool = Field(False, description="Whether the matching should be case-sensitive")

class CreateCollectionRequest(BaseModel):
    name: str = Field(..., description="Name of the collection")
    description: Optional[str] = Field(None, description="Description of the collection")

class AddToCollectionRequest(BaseModel):
    collection_id: str = Field(..., description="ID of the collection")
    effect_ids: List[str] = Field(..., description="List of sound effect IDs to add")

class ApplySpatialPositioningRequest(BaseModel):
    file_path: str = Field(..., description="Path to the sound effect file")
    output_path: str = Field(..., description="Path to save the processed file")
    position: Dict[str, float] = Field(..., description="3D position (x,y,z between -1.0 and 1.0)")
    intensity: float = Field(1.0, description="Volume intensity (0.0 to 1.0)")

class AdjustIntensityRequest(BaseModel):
    file_path: str = Field(..., description="Path to the sound effect file")
    output_path: str = Field(..., description="Path to save the processed file")
    intensity: float = Field(..., description="Volume intensity (0.0 to 1.0)")
    fade_in: Optional[float] = Field(None, description="Fade in duration in seconds")
    fade_out: Optional[float] = Field(None, description="Fade out duration in seconds")

class ProcessAndApplySoundEffectRequest(BaseModel):
    file_path: str = Field(..., description="Path to the sound effect file")
    output_path: str = Field(..., description="Path to save the processed file")
    target_file: Optional[str] = Field(None, description="Path to the target file to mix with")
    start_time: Optional[float] = Field(None, description="Start time in seconds to add the effect")
    intensity: float = Field(1.0, description="Volume intensity (0.0 to 1.0)")
    position: Optional[Dict[str, float]] = Field(None, description="Optional spatial position")

class ContextAwareRecommendationRequest(BaseModel):
    text: str = Field(..., description="Transcript or text description of the scene")
    video_path: Optional[str] = Field(None, description="Optional path to video file for scene analysis")
    categories: Optional[List[str]] = Field(None, description="Optional categories to filter by")
    max_results: int = Field(5, description="Maximum number of recommended sound effects")

# Sound Effects API Models
class SoundEffectRecommendationRequest(BaseModel):
    transcript: Optional[str] = Field(None, description="Text transcript of the video")
    scene_descriptions: Optional[List[str]] = Field(None, description="List of scene descriptions")
    video_category: Optional[str] = Field(None, description="Category of the video")
    mood: Optional[str] = Field(None, description="Detected mood of the video or scene")
    keywords: Optional[List[str]] = Field(None, description="Extracted keywords from the content")
    timeline_position: Optional[float] = Field(None, description="Position in the video timeline (in seconds)")
    intensity: Optional[float] = Field(0.5, description="Detected intensity of the scene (0.0 to 1.0)", ge=0.0, le=1.0)
    preceding_effects: Optional[List[str]] = Field(None, description="IDs of sound effects used before this point")
    max_results: Optional[int] = Field(10, description="Maximum number of results to return", ge=1, le=50)
    include_details: Optional[bool] = Field(True, description="Whether to include complete effect details")

@router.post("/sfx/recommend", response_model=Dict[str, Any], tags=["Sound Effects"])
async def recommend_sound_effects(
    request: SoundEffectRecommendationRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Get context-aware sound effect recommendations based on video content analysis.
    
    This endpoint analyzes the provided context (transcript, scene descriptions, mood, etc.)
    and recommends sound effects that would best match the content. The recommendation
    algorithm considers semantic relevance, category matching, mood congruence, and
    intensity appropriateness.
    
    - **Context Information**: Provide as much context as possible for better recommendations
    - **Preceding Effects**: Include previously used effects to avoid repetition
    - **Intensity**: Values range from 0.0 (subtle) to 1.0 (intense)
    """
    try:
        # Create library instance
        sfx_library = SoundEffectsLibrary()
        
        # Prepare context dictionary
        context = {
            "transcript": request.transcript or "",
            "scene_descriptions": request.scene_descriptions or [],
            "video_category": request.video_category or "",
            "mood": request.mood or "",
            "keywords": request.keywords or [],
            "timeline_position": request.timeline_position,
            "intensity": request.intensity,
            "preceding_effects": request.preceding_effects or []
        }
        
        # Get recommendations
        recommendations = sfx_library.recommend_sound_effects(
            context=context,
            max_results=request.max_results,
            include_details=request.include_details
        )
        
        return recommendations
    except Exception as e:
        logger.error(f"Error in sound effect recommendation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate sound effect recommendations: {str(e)}"
        )

# Sound Effects Library models
class SoundEffectMetadata(BaseModel):
    name: str = Field(..., description="Name of the sound effect")
    category: str = Field(..., description="Category of the sound effect")
    tags: List[str] = Field(default=[], description="Tags for the sound effect")
    description: Optional[str] = Field(None, description="Description of the sound effect")
    duration: Optional[float] = Field(None, description="Duration of the sound effect in seconds")
    file_format: Optional[str] = Field(None, description="File format of the sound effect")
    sample_rate: Optional[int] = Field(None, description="Sample rate of the sound effect")
    channels: Optional[int] = Field(None, description="Number of audio channels")
    
class AddSoundEffectRequest(BaseModel):
    name: str = Field(..., description="Name of the sound effect")
    category: str = Field(..., description="Category of the sound effect")
    tags: List[str] = Field(default=[], description="Tags for the sound effect")
    description: Optional[str] = Field(None, description="Description of the sound effect")
    spatial_data: Optional[Dict[str, Any]] = Field(None, description="Spatial audio positioning data")
    intensity_levels: Optional[Dict[str, float]] = Field(None, description="Different intensity levels for the sound effect")
    trigger_words: Optional[List[str]] = Field(None, description="Words that can trigger this sound effect")
    custom_library_id: Optional[str] = Field(None, description="ID of a custom library this effect belongs to")

class GetSoundEffectsRequest(BaseModel):
    category: Optional[str] = Field(None, description="Filter by category")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    search_term: Optional[str] = Field(None, description="Search in name and description")
    limit: int = Field(10, description="Maximum number of sound effects to return")
    offset: int = Field(0, description="Offset for pagination")

# Sound Effects Library endpoints
@router.post("/sound-effects/add", 
            response_model=Dict[str, Any],
            summary="Add a sound effect to the library",
            description="Upload and add a sound effect to the library with metadata")
async def add_sound_effect(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    metadata: str = Form(...),
    current_user: User = Depends(get_current_user)
):
    try:
        # Parse metadata
        metadata_dict = json.loads(metadata)
        add_request = AddSoundEffectRequest(**metadata_dict)
        
        # Save file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Initialize sound effects library
        sound_effects_library = SoundEffectsLibrary()
        
        # Add sound effect to library
        result = sound_effects_library.add_sound_effect(
            file_path=temp_file_path,
            name=add_request.name,
            category=add_request.category,
            tags=add_request.tags,
            description=add_request.description,
            spatial_data=add_request.spatial_data,
            intensity_levels=add_request.intensity_levels,
            trigger_words=add_request.trigger_words,
            custom_library_id=add_request.custom_library_id
        )
        
        # Cleanup temp file in background
        background_tasks.add_task(lambda: os.unlink(temp_file_path) if os.path.exists(temp_file_path) else None)
        background_tasks.add_task(lambda: os.rmdir(temp_dir) if os.path.exists(temp_dir) else None)
        
        return {
            "status": "success",
            "message": f"Sound effect '{add_request.name}' added to category '{add_request.category}'",
            "effect_id": result.get("effect_id"),
            "category": add_request.category
        }
    
    except Exception as e:
        logger.error(f"Error adding sound effect: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding sound effect: {str(e)}")

@router.get("/sound-effects/categories",
          response_model=Dict[str, Any],
          summary="Get all sound effect categories",
          description="Get all available sound effect categories with counts")
async def get_sound_effect_categories(
    current_user: User = Depends(get_current_user)
):
    try:
        sound_effects_library = SoundEffectsLibrary()
        categories = sound_effects_library.get_categories()
        
        return {
            "status": "success",
            "categories": categories
        }
    
    except Exception as e:
        logger.error(f"Error retrieving sound effect categories: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving sound effect categories: {str(e)}")

@router.post("/sound-effects/search",
           response_model=Dict[str, Any],
           summary="Search sound effects",
           description="Search for sound effects by category, tags, or text search")
async def search_sound_effects(
    request: GetSoundEffectsRequest,
    current_user: User = Depends(get_current_user)
):
    try:
        sound_effects_library = SoundEffectsLibrary()
        
        # Search sound effects
        sound_effects = sound_effects_library.search_sound_effects(
            category=request.category,
            tags=request.tags,
            search_term=request.search_term,
            limit=request.limit,
            offset=request.offset
        )
        
        return {
            "status": "success",
            "total_count": len(sound_effects),
            "sound_effects": sound_effects
        }
    
    except Exception as e:
        logger.error(f"Error searching sound effects: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching sound effects: {str(e)}")