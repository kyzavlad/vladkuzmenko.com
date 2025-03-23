"""
B-Roll API Endpoints

This module provides REST API endpoints for the B-Roll Insertion Engine.
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

from app.services.broll.broll_engine import BRollEngine
from app.core.config import settings
from app.core.auth import get_current_user, User

router = APIRouter()
logger = logging.getLogger(__name__)

# Pydantic models for API
class TranscriptSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str

class Transcript(BaseModel):
    segments: List[TranscriptSegment]

class BRollRequest(BaseModel):
    transcript: Transcript
    provider: Optional[str] = None
    max_suggestions: int = Field(3, ge=1, le=10)
    use_face_detection: bool = True
    use_audio_analysis: bool = True
    generate_preview: bool = True
    b_roll_density: Optional[float] = Field(None, ge=0.0, le=1.0)
    scene_detection_threshold: Optional[float] = Field(None, ge=5.0, le=50.0)
    use_semantic_matching: bool = True

class BRollAnalyzeRequest(BaseModel):
    transcript: Transcript
    provider: Optional[str] = None
    max_suggestions: int = Field(3, ge=1, le=10)

class BRollJobResponse(BaseModel):
    job_id: str
    status: str
    video_path: str
    output_dir: str
    created_at: float

# In-memory storage for job status (in a real app, use a database)
broll_jobs = {}

def get_engine_config(request: dict) -> dict:
    """Get configuration for the B-Roll Engine based on request parameters."""
    return {
        'content_analyzer': {
            'use_advanced_nlp': True,
        },
        'scene_detector': {
            'min_scene_length': 2.0,
            'use_face_detection': request.get('use_face_detection', True),
            'use_audio_analysis': request.get('use_audio_analysis', True),
            'detection_threshold': request.get('scene_detection_threshold', 30.0),
        },
        'stock_provider': {
            'api_keys': {
                'pexels': settings.pexels_api_key,
                'pixabay': settings.pixabay_api_key,
            },
            'user_library_paths': [
                settings.broll_library_path,
            ],
            'supported_providers': ['pexels', 'pixabay', 'local'],
        },
        'semantic_matcher': {
            'similarity_threshold': 0.65,
            'spacy_model': 'en_core_web_md',
            'concept_database_path': os.path.join(settings.broll_library_path, 'concept_database.json'),
        },
        'b_roll_density': request.get('b_roll_density', 0.3),
    }

def get_temp_path(prefix: str) -> str:
    """Get a temporary path."""
    # Create a temporary directory within the configured temp path
    os.makedirs(settings.temp_path, exist_ok=True)
    temp_dir = os.path.join(settings.temp_path, f"{prefix}_{int(time.time())}")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

async def process_broll_job(job_id: str, video_path: str, request_data: dict, output_dir: str):
    """
    Process a B-Roll job in the background.
    
    Args:
        job_id: Unique job identifier
        video_path: Path to the video file
        request_data: B-Roll request data
        output_dir: Directory to save outputs
    """
    try:
        # Update job status
        broll_jobs[job_id]['status'] = 'processing'
        
        # Initialize B-Roll Engine
        config = get_engine_config(request_data)
        config['output_dir'] = output_dir
        
        engine = BRollEngine(config)
        
        # Convert transcript format
        transcript = {
            'segments': [
                {
                    'id': segment.id,
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text
                }
                for segment in request_data['transcript'].segments
            ]
        }
        
        # Process options
        options = {
            'output_dir': output_dir,
            'max_suggestions': request_data.get('max_suggestions', 3),
            'provider': request_data.get('provider'),
            'generate_preview': request_data.get('generate_preview', True),
            'use_semantic_matching': request_data.get('use_semantic_matching', True),
        }
        
        # Process video
        results = await engine.process_video(video_path, transcript, options)
        
        # Update job with results
        broll_jobs[job_id].update({
            'status': 'completed',
            'completed_at': time.time(),
            'results': {
                'preview_path': results.get('preview_path'),
                'analysis_results': results.get('analysis_results'),
                'edit_suggestions': results.get('edit_suggestions'),
                'semantic_insights': results.get('suggestions', {}).get('semantic_insights', {}),
                'output_dir': output_dir
            }
        })
        
        # Close the engine
        await engine.close()
        
    except Exception as e:
        logger.error(f"Error processing B-Roll job {job_id}: {str(e)}")
        broll_jobs[job_id].update({
            'status': 'failed',
            'error': str(e)
        })
        
        # Clean up
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)

@router.post("/upload", response_class=JSONResponse)
async def upload_video(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """
    Upload a video file for B-Roll processing.
    
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

@router.post("/process", response_model=BRollJobResponse)
async def process_video(
    background_tasks: BackgroundTasks,
    request: BRollRequest,
    video_path: str = Query(..., description="Path to the uploaded video file"),
    current_user: User = Depends(get_current_user)
):
    """
    Process a video to suggest and insert B-Roll footage.
    
    Args:
        request: B-Roll processing request
        video_path: Path to the uploaded video file
        
    Returns:
        Job status information
    """
    # Validate video path
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    # Create output directory
    output_dir = get_temp_path("broll_output")
    
    # Create a unique job ID
    job_id = f"broll_{int(time.time())}_{os.path.basename(video_path)}"
    
    # Store job info
    broll_jobs[job_id] = {
        'job_id': job_id,
        'status': 'queued',
        'video_path': video_path,
        'output_dir': output_dir,
        'created_at': time.time()
    }
    
    # Start processing in the background
    request_data = request.dict()
    background_tasks.add_task(process_broll_job, job_id, video_path, request_data, output_dir)
    
    return broll_jobs[job_id]

@router.get("/jobs/{job_id}", response_class=JSONResponse)
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get the status of a B-Roll processing job.
    
    Args:
        job_id: Job identifier
        
    Returns:
        Job status information
    """
    if job_id not in broll_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return broll_jobs[job_id]

@router.get("/jobs/{job_id}/preview", response_class=FileResponse)
async def get_job_preview(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get the preview video for a completed B-Roll job.
    
    Args:
        job_id: Job identifier
        
    Returns:
        Preview video file
    """
    if job_id not in broll_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = broll_jobs[job_id]
    
    if job.get('status') != 'completed':
        raise HTTPException(status_code=400, detail="Job not completed")
    
    preview_path = job.get('results', {}).get('preview_path')
    
    if not preview_path or not os.path.exists(preview_path):
        raise HTTPException(status_code=404, detail="Preview not found")
    
    return FileResponse(preview_path)

@router.post("/analyze", response_class=JSONResponse)
async def analyze_content(
    request: BRollAnalyzeRequest,
    video_path: str = Query(..., description="Path to the uploaded video file"),
    current_user: User = Depends(get_current_user)
):
    """
    Analyze a video to identify B-Roll opportunities without generating the preview.
    
    Args:
        request: B-Roll analysis request
        video_path: Path to the uploaded video file
        
    Returns:
        Analysis results
    """
    # Validate video path
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    # Create output directory
    output_dir = get_temp_path("broll_analysis")
    
    # Initialize B-Roll Engine
    config = get_engine_config(request.dict())
    config['output_dir'] = output_dir
    
    engine = BRollEngine(config)
    
    try:
        # Convert transcript format
        transcript = {
            'segments': [
                {
                    'id': segment.id,
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text
                }
                for segment in request.transcript.segments
            ]
        }
        
        # Analyze video content
        analysis_results = await engine.analyze_video_content(video_path, transcript)
        
        # Generate suggestions
        suggestions = await engine.generate_b_roll_suggestions(
            analysis_results,
            max_suggestions=request.max_suggestions,
            provider=request.provider
        )
        
        # Generate edit suggestions
        edit_suggestions = await engine.generate_edit_suggestions(
            analysis_results,
            suggestions
        )
        
        # Prepare response
        response = {
            'content_analysis': analysis_results.get('content_analysis', {}),
            'scene_analysis': {
                'scenes': analysis_results.get('scene_analysis', {}).get('scenes', []),
                'insertion_points': analysis_results.get('scene_analysis', {}).get('insertion_points', [])
            },
            'suggestions': {
                'b_roll_count': edit_suggestions.get('b_roll_count', 0),
                'b_roll_time': edit_suggestions.get('b_roll_time', 0),
                'edl': edit_suggestions.get('edl', []),
            }
        }
        
        return response
    
    finally:
        # Close the engine
        await engine.close()
        
        # Clean up
        shutil.rmtree(output_dir, ignore_errors=True) 