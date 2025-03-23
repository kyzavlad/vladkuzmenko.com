"""
Face Tracking API Module

This module provides API routes for the face tracking functionality
used by the Clip Generation Service.
"""

import os
import cv2
import uuid
import tempfile
import shutil
import numpy as np
import logging
from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import time
from datetime import datetime

from app.clip_generation.services.face_tracking_manager import FaceTrackingManager
from app.clip_generation.services.smart_framing import SmartFraming
from app.clip_generation.config.settings import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(
    prefix="/face-tracking",
    tags=["face-tracking"],
    responses={404: {"description": "Not found"}},
)

# Load settings
settings = Settings()

# Global face tracking components
face_tracker = None
smart_framing = None


class FaceDetectionRequest(BaseModel):
    """Request model for face detection."""
    detection_threshold: float = Field(0.5, description="Confidence threshold for face detection")
    recognition_enabled: bool = Field(False, description="Enable face recognition")
    max_faces: int = Field(10, description="Maximum number of faces to detect")
    include_landmarks: bool = Field(True, description="Include facial landmarks in response")
    device: str = Field("cpu", description="Device to use for inference (cpu, cuda)")


class SmartFramingRequest(BaseModel):
    """Request model for smart framing."""
    width: int = Field(1280, description="Target width")
    height: int = Field(720, description="Target height")
    rule_of_thirds: bool = Field(True, description="Apply rule of thirds composition")
    smooth_factor: float = Field(0.8, description="Smoothing factor (0-1)")
    preserve_context: bool = Field(True, description="Preserve scene context")
    face_zoom_factor: float = Field(1.5, description="Zoom factor for face framing")


class VideoProcessRequest(BaseModel):
    """Request model for video processing with face tracking."""
    detection_interval: int = Field(5, description="Run detection every N frames")
    recognition_interval: int = Field(10, description="Run recognition every N frames")
    detection_threshold: float = Field(0.5, description="Confidence threshold for face detection")
    output_fps: Optional[float] = Field(None, description="Output FPS (default: source video FPS)")
    output_format: str = Field("mp4", description="Output video format")
    framing_config: SmartFramingRequest = Field(
        default_factory=SmartFramingRequest,
        description="Smart framing configuration"
    )


class FaceTrackingResponse(BaseModel):
    """Response model for face tracking."""
    task_id: str = Field(..., description="Tracking task ID")
    status: str = Field(..., description="Task status")
    message: str = Field(..., description="Status message")
    created_at: str = Field(..., description="Task creation timestamp")
    result_path: Optional[str] = Field(None, description="Path to result video")
    processing_stats: Optional[Dict[str, Any]] = Field(None, description="Processing statistics")


def get_face_tracker():
    """Get or initialize the face tracking manager."""
    global face_tracker
    if face_tracker is None:
        model_dir = os.path.join(settings.models_path, "face")
        face_tracker = FaceTrackingManager(
            model_dir=model_dir,
            detection_interval=5,
            recognition_interval=10,
            detection_threshold=0.5,
            recognition_threshold=0.6,
            max_faces=10,
            device="cpu"
        )
        logger.info("Initialized global face tracking manager")
    return face_tracker


def get_smart_framing():
    """Get or initialize the smart framing component."""
    global smart_framing
    if smart_framing is None:
        smart_framing = SmartFraming(
            target_width=1280,
            target_height=720,
            smoothing_factor=0.8,
            padding_ratio=0.1,
            rule_of_thirds=True,
            preserve_context=True
        )
        logger.info("Initialized global smart framing component")
    return smart_framing


def cleanup_temp_files(file_path: str):
    """Clean up temporary files."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Removed temporary file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up file {file_path}: {e}")


@router.post("/detect", response_model=Dict[str, Any])
async def detect_faces(
    image: UploadFile = File(...),
    config: Optional[FaceDetectionRequest] = None
):
    """
    Detect faces in an uploaded image.
    
    Args:
        image: Image file to analyze
        config: Face detection configuration
    
    Returns:
        Dictionary with detected faces information
    """
    if config is None:
        config = FaceDetectionRequest()
    
    # Create temp file for the image
    temp_dir = tempfile.mkdtemp()
    try:
        temp_path = os.path.join(temp_dir, image.filename)
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(image.file, f)
        
        # Read image
        frame = cv2.imread(temp_path)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Get face tracker with specified config
        tracker = get_face_tracker()
        tracker.detection_threshold = config.detection_threshold
        
        # Process frame
        start_time = time.time()
        tracked_faces = tracker.process_frame(frame)
        processing_time = time.time() - start_time
        
        # Prepare response
        height, width = frame.shape[:2]
        
        faces_data = []
        for face_id, face in tracked_faces.items():
            face_data = {
                "face_id": face_id,
                "bbox": {
                    "x1": float(face.box.x1),
                    "y1": float(face.box.y1),
                    "x2": float(face.box.x2),
                    "y2": float(face.box.y2),
                    "width": float(face.box.width),
                    "height": float(face.box.height)
                },
                "confidence": face.avg_confidence(),
                "is_speaker": face.is_speaker
            }
            
            # Add landmarks if requested and available
            if config.include_landmarks and face.box.landmarks is not None:
                face_data["landmarks"] = [
                    {"x": float(x), "y": float(y)} for x, y in face.box.landmarks
                ]
            
            # Add identity if available
            if face.identity is not None:
                face_data["identity"] = {
                    "name": face.identity.name,
                    "confidence": float(face.identity.confidence) if hasattr(face.identity, "confidence") else None
                }
            
            faces_data.append(face_data)
        
        # Reset tracker for next request
        tracker.reset()
        
        return {
            "faces": faces_data,
            "count": len(faces_data),
            "speaker_id": tracker.speaker_id,
            "image_size": {"width": width, "height": height},
            "processing_time_ms": processing_time * 1000
        }
    
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir)


@router.post("/process-video", response_model=FaceTrackingResponse)
async def process_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    config: Optional[VideoProcessRequest] = None
):
    """
    Process a video with face tracking and smart framing.
    
    Args:
        background_tasks: FastAPI background tasks
        video: Video file to process
        config: Processing configuration
    
    Returns:
        Response with task ID and status
    """
    if config is None:
        config = VideoProcessRequest()
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    # Create temp directory for processing
    temp_dir = os.path.join(settings.temp_dir, f"face_tracking_{task_id}")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save uploaded video
    input_path = os.path.join(temp_dir, f"input_{task_id}.mp4")
    with open(input_path, "wb") as f:
        shutil.copyfileobj(video.file, f)
    
    # Output path
    output_filename = f"tracked_{task_id}.{config.output_format}"
    output_path = os.path.join(settings.output_dir, output_filename)
    os.makedirs(settings.output_dir, exist_ok=True)
    
    # Create task response before processing
    response = FaceTrackingResponse(
        task_id=task_id,
        status="processing",
        message="Video processing started",
        created_at=timestamp,
        result_path=None,
        processing_stats=None
    )
    
    # Process video in background
    background_tasks.add_task(
        process_video_task,
        input_path=input_path,
        output_path=output_path,
        temp_dir=temp_dir,
        config=config,
        task_id=task_id
    )
    
    return response


@router.get("/task/{task_id}", response_model=FaceTrackingResponse)
async def get_task_status(task_id: str):
    """
    Get status of a video processing task.
    
    Args:
        task_id: Task ID
    
    Returns:
        Task status information
    """
    # Check if result file exists
    output_path = os.path.join(settings.output_dir, f"tracked_{task_id}.mp4")
    
    # Check if stats file exists
    stats_path = os.path.join(settings.output_dir, f"stats_{task_id}.json")
    
    if os.path.exists(output_path):
        # Task completed
        stats = None
        if os.path.exists(stats_path):
            try:
                import json
                with open(stats_path, "r") as f:
                    stats = json.load(f)
            except Exception as e:
                logger.error(f"Error loading stats for task {task_id}: {e}")
        
        return FaceTrackingResponse(
            task_id=task_id,
            status="completed",
            message="Video processing completed",
            created_at=datetime.fromtimestamp(os.path.getctime(output_path)).isoformat(),
            result_path=output_path,
            processing_stats=stats
        )
    
    # Check if task is in progress (temp dir exists)
    temp_dir = os.path.join(settings.temp_dir, f"face_tracking_{task_id}")
    if os.path.exists(temp_dir):
        return FaceTrackingResponse(
            task_id=task_id,
            status="processing",
            message="Video processing in progress",
            created_at=datetime.fromtimestamp(os.path.getctime(temp_dir)).isoformat(),
            result_path=None,
            processing_stats=None
        )
    
    # Task not found
    raise HTTPException(status_code=404, detail=f"Task {task_id} not found")


@router.get("/result/{task_id}")
async def get_result_video(task_id: str):
    """
    Download the result video for a completed task.
    
    Args:
        task_id: Task ID
    
    Returns:
        Video file response
    """
    output_path = os.path.join(settings.output_dir, f"tracked_{task_id}.mp4")
    
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail=f"Result for task {task_id} not found")
    
    return FileResponse(output_path, media_type="video/mp4", filename=f"tracked_{task_id}.mp4")


@router.delete("/task/{task_id}")
async def delete_task(task_id: str, background_tasks: BackgroundTasks):
    """
    Delete a task and its associated files.
    
    Args:
        task_id: Task ID
        background_tasks: FastAPI background tasks
    
    Returns:
        Deletion status
    """
    # Check if task exists
    output_path = os.path.join(settings.output_dir, f"tracked_{task_id}.mp4")
    temp_dir = os.path.join(settings.temp_dir, f"face_tracking_{task_id}")
    stats_path = os.path.join(settings.output_dir, f"stats_{task_id}.json")
    
    if not os.path.exists(output_path) and not os.path.exists(temp_dir):
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    # Delete files in background
    background_tasks.add_task(delete_task_files, task_id, output_path, temp_dir, stats_path)
    
    return {"status": "success", "message": f"Task {task_id} deletion initiated"}


async def process_video_task(
    input_path: str,
    output_path: str,
    temp_dir: str,
    config: VideoProcessRequest,
    task_id: str
):
    """
    Process video with face tracking and smart framing (background task).
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        temp_dir: Temporary directory for processing
        config: Processing configuration
        task_id: Task ID
    """
    stats = {
        "task_id": task_id,
        "started_at": datetime.now().isoformat(),
        "frames_processed": 0,
        "faces_detected": 0,
        "processing_time_seconds": 0
    }
    
    try:
        # Initialize face tracker
        model_dir = os.path.join(settings.models_path, "face")
        face_tracker = FaceTrackingManager(
            model_dir=model_dir,
            detection_interval=config.detection_interval,
            recognition_interval=config.recognition_interval,
            detection_threshold=config.detection_threshold,
            device="cpu"  # TODO: support GPU if available
        )
        
        # Initialize smart framing
        framing_config = config.framing_config
        smart_framing = SmartFraming(
            target_width=framing_config.width,
            target_height=framing_config.height,
            smoothing_factor=framing_config.smooth_factor,
            padding_ratio=0.1,
            rule_of_thirds=framing_config.rule_of_thirds,
            preserve_context=framing_config.preserve_context,
            face_zoom_factor=framing_config.face_zoom_factor
        )
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception(f"Failed to open video: {input_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Use source FPS if not specified
        output_fps = config.output_fps if config.output_fps is not None else fps
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            output_fps,
            (framing_config.width, framing_config.height)
        )
        
        # Process frames
        frame_idx = 0
        face_count = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame with face tracking
            tracked_faces = face_tracker.process_frame(frame)
            face_count += len(tracked_faces)
            
            # Get current speaker
            speaker_id = face_tracker.speaker_id
            
            # Generate smart framing
            framed_image, _ = smart_framing.frame_image(
                frame,
                tracked_faces,
                speaker_id
            )
            
            # Write output frame
            out.write(framed_image)
            
            # Update frame count
            frame_idx += 1
            
            # Log progress
            if frame_idx % 100 == 0:
                progress = frame_idx / total_frames * 100
                logger.info(f"Task {task_id}: Processed {frame_idx}/{total_frames} frames ({progress:.1f}%)")
        
        # Clean up
        cap.release()
        out.release()
        
        # Calculate statistics
        processing_time = time.time() - start_time
        avg_faces_per_frame = face_count / frame_idx if frame_idx > 0 else 0
        
        # Update stats
        stats.update({
            "completed_at": datetime.now().isoformat(),
            "frames_processed": frame_idx,
            "total_frames": total_frames,
            "faces_detected": face_count,
            "avg_faces_per_frame": avg_faces_per_frame,
            "processing_time_seconds": processing_time,
            "fps": frame_idx / processing_time if processing_time > 0 else 0,
            "input_resolution": f"{width}x{height}",
            "output_resolution": f"{framing_config.width}x{framing_config.height}",
            "input_fps": fps,
            "output_fps": output_fps,
            "status": "completed"
        })
        
        # Save stats
        stats_path = os.path.join(settings.output_dir, f"stats_{task_id}.json")
        import json
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Task {task_id}: Video processing completed successfully")
    
    except Exception as e:
        logger.error(f"Task {task_id}: Error processing video: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Update stats with error
        stats.update({
            "completed_at": datetime.now().isoformat(),
            "status": "failed",
            "error": str(e)
        })
        
        # Save stats
        stats_path = os.path.join(settings.output_dir, f"stats_{task_id}.json")
        import json
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
    
    finally:
        # Clean up temp directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.error(f"Task {task_id}: Error cleaning up temp directory: {e}")


async def delete_task_files(
    task_id: str,
    output_path: str,
    temp_dir: str,
    stats_path: str
):
    """
    Delete all files associated with a task.
    
    Args:
        task_id: Task ID
        output_path: Path to output video
        temp_dir: Path to temporary directory
        stats_path: Path to stats file
    """
    try:
        # Delete output file
        if os.path.exists(output_path):
            os.remove(output_path)
            logger.info(f"Deleted output file for task {task_id}")
        
        # Delete temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Deleted temp directory for task {task_id}")
        
        # Delete stats file
        if os.path.exists(stats_path):
            os.remove(stats_path)
            logger.info(f"Deleted stats file for task {task_id}")
    
    except Exception as e:
        logger.error(f"Error deleting files for task {task_id}: {e}")
        import traceback
        logger.error(traceback.format_exc()) 