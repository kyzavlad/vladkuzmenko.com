"""
Avatar API Endpoints

This module provides API endpoints for avatar generation functionality,
including 3D face reconstruction, texture mapping, and avatar creation.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends, Query
from typing import Dict, Any, List, Optional, Union
import os
import logging
import uuid
import json
from pathlib import Path
import tempfile
import shutil
from pydantic import BaseModel, Field

from app.services.avatar.face_modeling import FaceModeling
from app.services.avatar.animation_framework import AnimationFramework
from app.api.deps import get_storage_service
from app.services.storage_service import StorageService
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize the FaceModeling service
face_modeling = FaceModeling()

# Initialize the AnimationFramework service
animation_framework = AnimationFramework()

# Request and response models
class Face3DReconstructionRequest(BaseModel):
    """Request model for 3D face reconstruction from an image."""
    detail_level: str = Field("high", description="Detail level for face reconstruction (low, medium, high, ultra)")
    enable_texture_mapping: bool = Field(True, description="Enable high-fidelity 4K texture mapping")
    enable_detail_refinement: bool = Field(True, description="Enable detailed feature preservation algorithm")
    enable_identity_verification: bool = Field(True, description="Enable identity consistency verification")
    enable_stylegan_enhancements: bool = Field(True, description="Enable StyleGAN-3 enhancements")
    enable_expression_calibration: bool = Field(True, description="Enable expression range calibration")
    
class Face3DReconstructionResponse(BaseModel):
    """Response model for 3D face reconstruction."""
    model_id: str = Field(..., description="Unique ID for the generated 3D model")
    model_url: str = Field(..., description="URL to download the 3D model")
    texture_url: str = Field(..., description="URL to download the texture map")
    quality_score: float = Field(..., description="Quality score of the generated model (0-1)")
    processing_time: float = Field(..., description="Time taken to process the request in seconds")
    identity_verification_score: Optional[float] = Field(None, description="Identity verification score (0-1)")
    landmarks: Dict[str, List[float]] = Field(..., description="Detected facial landmarks")
    metadata: Dict[str, Any] = Field({}, description="Additional metadata about the model")

@router.post("/face/reconstruct-from-image", response_model=Face3DReconstructionResponse)
async def reconstruct_face_from_image(
    image: UploadFile = File(..., description="Image file containing a face"),
    options: Face3DReconstructionRequest = Depends(),
    storage_service: StorageService = Depends(get_storage_service)
):
    """
    Generate a 3D face model from a single image.
    
    This endpoint processes an uploaded image to create a high-fidelity 3D model
    of the face in the image, with texture mapping and expression capabilities.
    """
    # Create a temporary file to store the uploaded image
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename)[1]) as temp_image:
        # Write the uploaded file to the temporary file
        shutil.copyfileobj(image.file, temp_image)
        temp_image_path = temp_image.name
    
    try:
        # Prepare options for the face modeling service
        processing_options = {
            "detail_level": options.detail_level,
            "texture_mapping_enabled": options.enable_texture_mapping,
            "detail_refinement_enabled": options.enable_detail_refinement,
            "identity_verification_enabled": options.enable_identity_verification,
            "stylegan_enabled": options.enable_stylegan_enhancements,
            "expression_calibration_enabled": options.enable_expression_calibration
        }
        
        # Process the image and generate 3D face model
        result = await face_modeling.generate_from_image(temp_image_path, processing_options)
        
        # Store the generated model and texture in the storage service
        model_storage_path = f"avatars/{result.model_id}/model{os.path.splitext(result.model_path)[1]}"
        texture_storage_path = f"avatars/{result.model_id}/texture{os.path.splitext(result.texture_path)[1]}"
        
        await storage_service.upload_file(result.model_path, model_storage_path)
        await storage_service.upload_file(result.texture_path, texture_storage_path)
        
        # Generate URLs for the model and texture
        model_url = await storage_service.get_presigned_url(model_storage_path)
        texture_url = await storage_service.get_presigned_url(texture_storage_path)
        
        # Prepare and return the response
        return Face3DReconstructionResponse(
            model_id=result.model_id,
            model_url=model_url,
            texture_url=texture_url,
            quality_score=result.quality_score,
            processing_time=result.processing_time,
            identity_verification_score=result.identity_verification_score,
            landmarks=result.landmarks,
            metadata=result.metadata or {}
        )
        
    except Exception as e:
        logger.error(f"Error in 3D face reconstruction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Face reconstruction failed: {str(e)}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_image_path):
            os.unlink(temp_image_path)

@router.post("/face/reconstruct-from-video", response_model=Face3DReconstructionResponse)
async def reconstruct_face_from_video(
    video: UploadFile = File(..., description="Video file containing a face"),
    options: Face3DReconstructionRequest = Depends(),
    storage_service: StorageService = Depends(get_storage_service)
):
    """
    Generate a high-quality 3D face model from a video.
    
    This endpoint processes an uploaded video to create a high-fidelity 3D model
    of the face in the video, utilizing multiple frames for improved accuracy.
    """
    # Create a temporary file to store the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video.filename)[1]) as temp_video:
        # Write the uploaded file to the temporary file
        shutil.copyfileobj(video.file, temp_video)
        temp_video_path = temp_video.name
    
    try:
        # Prepare options for the face modeling service
        processing_options = {
            "detail_level": options.detail_level,
            "texture_mapping_enabled": options.enable_texture_mapping,
            "detail_refinement_enabled": options.enable_detail_refinement,
            "identity_verification_enabled": options.enable_identity_verification,
            "stylegan_enabled": options.enable_stylegan_enhancements,
            "expression_calibration_enabled": options.enable_expression_calibration
        }
        
        # Process the video and generate 3D face model
        result = await face_modeling.generate_from_video(temp_video_path, processing_options)
        
        # Store the generated model and texture in the storage service
        model_storage_path = f"avatars/{result.model_id}/model{os.path.splitext(result.model_path)[1]}"
        texture_storage_path = f"avatars/{result.model_id}/texture{os.path.splitext(result.texture_path)[1]}"
        
        await storage_service.upload_file(result.model_path, model_storage_path)
        await storage_service.upload_file(result.texture_path, texture_storage_path)
        
        # Generate URLs for the model and texture
        model_url = await storage_service.get_presigned_url(model_storage_path)
        texture_url = await storage_service.get_presigned_url(texture_storage_path)
        
        # Prepare and return the response
        return Face3DReconstructionResponse(
            model_id=result.model_id,
            model_url=model_url,
            texture_url=texture_url,
            quality_score=result.quality_score,
            processing_time=result.processing_time,
            identity_verification_score=result.identity_verification_score,
            landmarks=result.landmarks,
            metadata=result.metadata or {}
        )
        
    except Exception as e:
        logger.error(f"Error in 3D face reconstruction from video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Face reconstruction from video failed: {str(e)}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_video_path):
            os.unlink(temp_video_path)

# Animation Framework Endpoints

class AnimateAvatarRequest(BaseModel):
    """Request model for animating a 3D face model with a driving video."""
    enhance_micro_expressions: bool = Field(True, description="Enhance animation with natural micro-expressions")
    temporal_smoothness: float = Field(0.8, description="Smoothness factor for temporal consistency (0.0-1.0)", ge=0.0, le=1.0)
    smoothing_window: int = Field(5, description="Window size for temporal smoothing", ge=1, le=15)

class AnimationResponse(BaseModel):
    """Response model for avatar animation."""
    animation_id: str = Field(..., description="Unique ID for the generated animation")
    animation_url: str = Field(..., description="URL to download the animation")
    duration: float = Field(..., description="Duration of the animation in seconds")
    frame_count: int = Field(..., description="Number of frames in the animation")
    fps: float = Field(..., description="Frames per second of the animation")
    processing_time: float = Field(..., description="Time taken to process the request in seconds")
    metadata: Dict[str, Any] = Field({}, description="Additional metadata about the animation")

@router.post("/animate", response_model=AnimationResponse)
async def animate_avatar(
    model_file: UploadFile = File(..., description="3D face model file"),
    driving_video: UploadFile = File(..., description="Driving video file"),
    options: AnimateAvatarRequest = Depends(),
    storage_service: StorageService = Depends(get_storage_service)
):
    """
    Animate a 3D avatar using a driving video.
    
    This endpoint uses the First Order Motion Model with temporal consistency to animate
    a 3D face model based on the movements in a driving video. The animation includes
    68-point facial landmark tracking and natural micro-expression synthesis.
    """
    logger.info("Received request to animate avatar")
    
    # Create temporary directory to store uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded files to temporary directory
        model_path = os.path.join(temp_dir, f"model_{uuid.uuid4()}{os.path.splitext(model_file.filename)[1]}")
        video_path = os.path.join(temp_dir, f"video_{uuid.uuid4()}{os.path.splitext(driving_video.filename)[1]}")
        
        # Save model file
        with open(model_path, "wb") as f:
            f.write(await model_file.read())
            
        # Save driving video file
        with open(video_path, "wb") as f:
            f.write(await driving_video.read())
        
        try:
            # Convert Pydantic model to dict
            options_dict = options.dict()
            
            # Animate the avatar
            result = await animation_framework.animate_from_video(
                face_model_path=model_path,
                driving_video_path=video_path,
                options=options_dict
            )
            
            # Upload the animation to storage
            animation_storage_path = f"animations/{result.animation_id}{os.path.splitext(result.output_path)[1]}"
            animation_uploaded = await storage_service.upload_file(result.output_path, animation_storage_path)
            
            if not animation_uploaded:
                raise HTTPException(status_code=500, detail="Failed to upload animation file")
                
            # Get public URL for the animation
            animation_url = await storage_service.get_presigned_url(animation_storage_path)
            
            # Return response
            return AnimationResponse(
                animation_id=result.animation_id,
                animation_url=animation_url,
                duration=result.duration,
                frame_count=result.frame_count,
                fps=result.fps,
                processing_time=result.processing_time,
                metadata=result.metadata or {}
            )
            
        except Exception as e:
            logger.error(f"Error animating avatar: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error animating avatar: {str(e)}")
            
@router.post("/animate-from-video", response_model=AnimationResponse)
async def animate_avatar_from_video(
    face_id: str = Form(..., description="ID of the 3D face model to animate"),
    driving_video: UploadFile = File(..., description="Driving video file"),
    options: AnimateAvatarRequest = Depends(),
    storage_service: StorageService = Depends(get_storage_service)
):
    """
    Animate a previously generated 3D avatar using a driving video.
    
    This endpoint retrieves a previously generated 3D face model by ID and animates it
    using the First Order Motion Model with temporal consistency based on the movements
    in a driving video.
    """
    logger.info(f"Received request to animate avatar with ID: {face_id}")
    
    # Create temporary directory to store files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Construct paths for the model
        model_storage_path = f"avatars/{face_id}/model.glb"  # Assuming .glb format
        
        # Check if model exists in storage
        if not await storage_service.file_exists(model_storage_path):
            raise HTTPException(status_code=404, detail=f"3D face model with ID {face_id} not found")
            
        # Download model from storage
        model_path = os.path.join(temp_dir, f"model_{face_id}.glb")
        model_downloaded = await storage_service.download_file(model_storage_path, model_path)
        
        if not model_downloaded:
            raise HTTPException(status_code=500, detail=f"Failed to download 3D face model with ID {face_id}")
            
        # Save driving video file
        video_path = os.path.join(temp_dir, f"video_{uuid.uuid4()}{os.path.splitext(driving_video.filename)[1]}")
        with open(video_path, "wb") as f:
            f.write(await driving_video.read())
        
        try:
            # Convert Pydantic model to dict
            options_dict = options.dict()
            
            # Animate the avatar
            result = await animation_framework.animate_from_video(
                face_model_path=model_path,
                driving_video_path=video_path,
                options=options_dict
            )
            
            # Upload the animation to storage
            animation_storage_path = f"animations/{result.animation_id}{os.path.splitext(result.output_path)[1]}"
            animation_uploaded = await storage_service.upload_file(result.output_path, animation_storage_path)
            
            if not animation_uploaded:
                raise HTTPException(status_code=500, detail="Failed to upload animation file")
                
            # Get public URL for the animation
            animation_url = await storage_service.get_presigned_url(animation_storage_path)
            
            # Return response
            return AnimationResponse(
                animation_id=result.animation_id,
                animation_url=animation_url,
                duration=result.duration,
                frame_count=result.frame_count,
                fps=result.fps,
                processing_time=result.processing_time,
                metadata=result.metadata or {}
            )
            
        except Exception as e:
            logger.error(f"Error animating avatar: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error animating avatar: {str(e)}") 