"""
Sound Effects API Endpoints

This module provides API endpoints for the Sound Effects Library and Processor,
enabling integration with the video processing service.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel, Field, validator
from starlette.status import HTTP_404_NOT_FOUND, HTTP_400_BAD_REQUEST

from app.services.sound_effects.sound_effects_library import SoundEffectsLibrary
from app.services.sound_effects.sound_effects_processor import SoundEffectsProcessor
from app.api.deps import get_sound_effects_library, get_sound_effects_processor

router = APIRouter(prefix="/sound-effects")

logger = logging.getLogger(__name__)

# Request and Response Models
class SoundEffectMetadata(BaseModel):
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    file_format: Optional[str] = None
    bit_depth: Optional[int] = None
    file_size: Optional[int] = None

class SpatialData(BaseModel):
    position_x: float = Field(0.0, ge=-1.0, le=1.0, description="X position (-1.0 to 1.0)")
    position_y: float = Field(0.0, ge=-1.0, le=1.0, description="Y position (-1.0 to 1.0)")
    position_z: Optional[float] = Field(0.0, ge=-1.0, le=1.0, description="Z position for 3D audio")
    width: Optional[float] = Field(0.2, ge=0.0, le=1.0, description="Spatial width")

class IntensityLevels(BaseModel):
    low: float = Field(0.5, ge=0.0, le=1.0, description="Low intensity level")
    medium: float = Field(0.75, ge=0.0, le=1.0, description="Medium intensity level")
    high: float = Field(1.0, ge=0.0, le=1.0, description="High intensity level")

class AddSoundEffectRequest(BaseModel):
    file_path: str = Field(..., description="Path to the sound effect file")
    name: str = Field(..., description="Name of the sound effect")
    category: str = Field(..., description="Category of the sound effect")
    tags: Optional[List[str]] = Field(None, description="Tags for the sound effect")
    description: Optional[str] = Field(None, description="Description of the sound effect")
    metadata: Optional[SoundEffectMetadata] = Field(None, description="Technical metadata")
    spatial_data: Optional[SpatialData] = Field(None, description="Spatial positioning data")
    intensity_levels: Optional[IntensityLevels] = Field(None, description="Intensity adjustment levels")
    trigger_words: Optional[List[str]] = Field(None, description="Trigger words for semantic analysis")
    custom_library_id: Optional[str] = Field(None, description="ID of custom library if applicable")
    
    @validator('file_path')
    def validate_file_exists(cls, v):
        if not os.path.exists(v):
            raise ValueError(f"File does not exist: {v}")
        return v

class UpdateSoundEffectRequest(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    description: Optional[str] = None
    metadata: Optional[SoundEffectMetadata] = None
    spatial_data: Optional[SpatialData] = None
    intensity_levels: Optional[IntensityLevels] = None
    trigger_words: Optional[List[str]] = None

class SearchSoundEffectsRequest(BaseModel):
    query: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    min_duration: Optional[float] = None
    max_duration: Optional[float] = None
    max_results: int = Field(20, ge=1, le=100)
    collection_id: Optional[str] = None

class TriggerWordsRequest(BaseModel):
    text: str = Field(..., description="Text to analyze for trigger words")
    max_results_per_word: int = Field(3, ge=1, le=10)
    case_sensitive: bool = False

class CreateCollectionRequest(BaseModel):
    name: str
    description: Optional[str] = None
    tags: Optional[List[str]] = None

class UpdateCollectionRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None

class CollectionItemsRequest(BaseModel):
    effect_ids: List[str]

class SpatialPositioningRequest(BaseModel):
    input_file: str = Field(..., description="Path to input sound effect file")
    output_file: str = Field(..., description="Path to output file after processing")
    position: SpatialData
    channels: int = Field(2, ge=2, le=6)
    intensity: float = Field(1.0, ge=0.0, le=2.0)
    
    @validator('input_file', 'output_file')
    def validate_files(cls, v):
        if v.startswith('/'):
            if not os.path.exists(os.path.dirname(v)):
                raise ValueError(f"Directory does not exist: {os.path.dirname(v)}")
        return v

class IntensityAdjustmentRequest(BaseModel):
    input_file: str
    output_file: str
    intensity: float = Field(..., ge=0.0, le=2.0)
    fade_in: Optional[float] = Field(None, ge=0.0)
    fade_out: Optional[float] = Field(None, ge=0.0)

class AddToVideoRequest(BaseModel):
    video_file: str
    sound_file: str
    output_file: str
    start_time: float = Field(..., ge=0.0)
    intensity: float = Field(1.0, ge=0.0, le=2.0)
    position: Optional[SpatialData] = None
    keep_original_audio: bool = True

class SoundEffectMixItem(BaseModel):
    file_path: str
    start_time: float = 0.0
    intensity: float = Field(1.0, ge=0.0, le=2.0)
    position: Optional[SpatialData] = None

class MixMultipleRequest(BaseModel):
    sound_files: List[SoundEffectMixItem]
    output_file: str
    target_duration: Optional[float] = None
    master_volume: float = Field(1.0, ge=0.0, le=2.0)

class ApplyToVideoRequest(BaseModel):
    video_file: str
    output_file: str
    sound_effects: List[SoundEffectMixItem]
    keep_original_audio: bool = True
    master_volume: float = Field(1.0, ge=0.0, le=2.0)

class AnalyzeSceneRequest(BaseModel):
    video_file: str
    segment_duration: float = Field(1.0, ge=0.1, le=10.0)

class RecommendSoundEffectsRequest(BaseModel):
    transcript: Optional[str] = Field(None, description="Text transcript of the video")
    scene_descriptions: Optional[List[str]] = Field(None, description="List of scene descriptions")
    video_category: Optional[str] = Field(None, description="Category of the video")
    mood: Optional[str] = Field(None, description="Detected mood of the video or scene")
    keywords: Optional[List[str]] = Field(None, description="Extracted keywords from the content")
    timeline_position: Optional[float] = Field(None, description="Position in the video timeline (in seconds)")
    intensity: float = Field(0.5, ge=0.0, le=1.0, description="Detected intensity of the scene (0.0 to 1.0)")
    preceding_effects: Optional[List[str]] = Field(None, description="IDs of sound effects used before this point")
    max_results: int = Field(10, ge=1, le=50, description="Maximum number of results to return")
    include_details: bool = Field(True, description="Whether to include complete effect details")

    class Config:
        schema_extra = {
            "example": {
                "transcript": "The car speeds down the highway as sirens wail in the distance.",
                "scene_descriptions": ["High-speed car chase", "Police pursuit"],
                "video_category": "action",
                "mood": "tense",
                "keywords": ["car", "chase", "police", "speed"],
                "intensity": 0.8,
                "max_results": 5
            }
        }

# API Endpoints for Sound Effects Library
@router.post("/library/add", response_model=Dict[str, Any], tags=["sound_effects"])
async def add_sound_effect(
    request: AddSoundEffectRequest,
    sfx_library: SoundEffectsLibrary = Depends(get_sound_effects_library)
):
    """
    Add a new sound effect to the library.
    """
    try:
        result = sfx_library.add_sound_effect(
            file_path=request.file_path,
            name=request.name,
            category=request.category,
            tags=request.tags,
            description=request.description,
            metadata=request.metadata.dict() if request.metadata else None,
            spatial_data=request.spatial_data.dict() if request.spatial_data else None,
            intensity_levels=request.intensity_levels.dict() if request.intensity_levels else None,
            trigger_words=request.trigger_words,
            custom_library_id=request.custom_library_id
        )
        return result
    except Exception as e:
        logger.error(f"Error adding sound effect: {str(e)}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Failed to add sound effect: {str(e)}"
        )

@router.get("/library/get/{effect_id}", response_model=Dict[str, Any], tags=["sound_effects"])
async def get_sound_effect(
    effect_id: str,
    sfx_library: SoundEffectsLibrary = Depends(get_sound_effects_library)
):
    """
    Get a specific sound effect from the library.
    """
    try:
        result = sfx_library.get_sound_effect(effect_id)
        return result
    except Exception as e:
        logger.error(f"Error getting sound effect: {str(e)}")
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=f"Failed to get sound effect: {str(e)}"
        )

@router.put("/library/update/{effect_id}", response_model=Dict[str, Any], tags=["sound_effects"])
async def update_sound_effect(
    effect_id: str,
    request: UpdateSoundEffectRequest,
    sfx_library: SoundEffectsLibrary = Depends(get_sound_effects_library)
):
    """
    Update a sound effect in the library.
    """
    try:
        result = sfx_library.update_sound_effect(
            effect_id=effect_id,
            name=request.name,
            category=request.category,
            tags=request.tags,
            description=request.description,
            metadata=request.metadata.dict() if request.metadata else None,
            spatial_data=request.spatial_data.dict() if request.spatial_data else None,
            intensity_levels=request.intensity_levels.dict() if request.intensity_levels else None,
            trigger_words=request.trigger_words
        )
        return result
    except Exception as e:
        logger.error(f"Error updating sound effect: {str(e)}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Failed to update sound effect: {str(e)}"
        )

@router.delete("/library/delete/{effect_id}", response_model=Dict[str, Any], tags=["sound_effects"])
async def delete_sound_effect(
    effect_id: str,
    sfx_library: SoundEffectsLibrary = Depends(get_sound_effects_library)
):
    """
    Delete a sound effect from the library.
    """
    try:
        result = sfx_library.delete_sound_effect(effect_id)
        return result
    except Exception as e:
        logger.error(f"Error deleting sound effect: {str(e)}")
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=f"Failed to delete sound effect: {str(e)}"
        )

@router.post("/library/search", response_model=Dict[str, Any], tags=["sound_effects"])
async def search_sound_effects(
    request: SearchSoundEffectsRequest,
    sfx_library: SoundEffectsLibrary = Depends(get_sound_effects_library)
):
    """
    Search for sound effects in the library.
    """
    try:
        result = sfx_library.search_sound_effects(
            query=request.query,
            category=request.category,
            tags=request.tags,
            min_duration=request.min_duration,
            max_duration=request.max_duration,
            max_results=request.max_results,
            collection_id=request.collection_id
        )
        return result
    except Exception as e:
        logger.error(f"Error searching sound effects: {str(e)}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Failed to search sound effects: {str(e)}"
        )

@router.post("/library/trigger-words", response_model=Dict[str, Any], tags=["sound_effects"])
async def find_by_trigger_words(
    request: TriggerWordsRequest,
    sfx_library: SoundEffectsLibrary = Depends(get_sound_effects_library)
):
    """
    Find sound effects based on trigger words in text.
    """
    try:
        result = sfx_library.find_by_trigger_words(
            text=request.text,
            max_results_per_word=request.max_results_per_word,
            case_sensitive=request.case_sensitive
        )
        return result
    except Exception as e:
        logger.error(f"Error finding sound effects by trigger words: {str(e)}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Failed to find sound effects by trigger words: {str(e)}"
        )

@router.post("/library/collection/create", response_model=Dict[str, Any], tags=["sound_effects"])
async def create_collection(
    request: CreateCollectionRequest,
    sfx_library: SoundEffectsLibrary = Depends(get_sound_effects_library)
):
    """
    Create a new sound effect collection.
    """
    try:
        result = sfx_library.create_collection(
            name=request.name,
            description=request.description,
            tags=request.tags
        )
        return result
    except Exception as e:
        logger.error(f"Error creating collection: {str(e)}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Failed to create collection: {str(e)}"
        )

@router.put("/library/collection/update/{collection_id}", response_model=Dict[str, Any], tags=["sound_effects"])
async def update_collection(
    collection_id: str,
    request: UpdateCollectionRequest,
    sfx_library: SoundEffectsLibrary = Depends(get_sound_effects_library)
):
    """
    Update a sound effect collection.
    """
    try:
        result = sfx_library.update_collection(
            collection_id=collection_id,
            name=request.name,
            description=request.description,
            tags=request.tags
        )
        return result
    except Exception as e:
        logger.error(f"Error updating collection: {str(e)}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Failed to update collection: {str(e)}"
        )

@router.delete("/library/collection/delete/{collection_id}", response_model=Dict[str, Any], tags=["sound_effects"])
async def delete_collection(
    collection_id: str,
    sfx_library: SoundEffectsLibrary = Depends(get_sound_effects_library)
):
    """
    Delete a sound effect collection.
    """
    try:
        result = sfx_library.delete_collection(collection_id)
        return result
    except Exception as e:
        logger.error(f"Error deleting collection: {str(e)}")
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=f"Failed to delete collection: {str(e)}"
        )

@router.post("/library/collection/{collection_id}/add", response_model=Dict[str, Any], tags=["sound_effects"])
async def add_to_collection(
    collection_id: str,
    request: CollectionItemsRequest,
    sfx_library: SoundEffectsLibrary = Depends(get_sound_effects_library)
):
    """
    Add sound effects to a collection.
    """
    try:
        result = sfx_library.add_to_collection(
            collection_id=collection_id,
            effect_ids=request.effect_ids
        )
        return result
    except Exception as e:
        logger.error(f"Error adding to collection: {str(e)}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Failed to add to collection: {str(e)}"
        )

@router.post("/library/collection/{collection_id}/remove", response_model=Dict[str, Any], tags=["sound_effects"])
async def remove_from_collection(
    collection_id: str,
    request: CollectionItemsRequest,
    sfx_library: SoundEffectsLibrary = Depends(get_sound_effects_library)
):
    """
    Remove sound effects from a collection.
    """
    try:
        result = sfx_library.remove_from_collection(
            collection_id=collection_id,
            effect_ids=request.effect_ids
        )
        return result
    except Exception as e:
        logger.error(f"Error removing from collection: {str(e)}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Failed to remove from collection: {str(e)}"
        )

@router.get("/library/collection/{collection_id}", response_model=Dict[str, Any], tags=["sound_effects"])
async def get_collection(
    collection_id: str,
    sfx_library: SoundEffectsLibrary = Depends(get_sound_effects_library)
):
    """
    Get a sound effect collection.
    """
    try:
        result = sfx_library.get_collection(collection_id)
        return result
    except Exception as e:
        logger.error(f"Error getting collection: {str(e)}")
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=f"Failed to get collection: {str(e)}"
        )

@router.get("/library/collections", response_model=Dict[str, Any], tags=["sound_effects"])
async def get_all_collections(
    sfx_library: SoundEffectsLibrary = Depends(get_sound_effects_library)
):
    """
    Get all sound effect collections.
    """
    try:
        result = sfx_library.get_all_collections()
        return result
    except Exception as e:
        logger.error(f"Error getting all collections: {str(e)}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Failed to get all collections: {str(e)}"
        )

@router.get("/library/categories", response_model=Dict[str, Any], tags=["sound_effects"])
async def get_all_categories(
    sfx_library: SoundEffectsLibrary = Depends(get_sound_effects_library)
):
    """
    Get all sound effect categories.
    """
    try:
        result = sfx_library.get_all_categories()
        return result
    except Exception as e:
        logger.error(f"Error getting categories: {str(e)}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Failed to get categories: {str(e)}"
        )

@router.get("/library/effects", response_model=Dict[str, Any], tags=["sound_effects"])
async def get_all_sound_effects(
    limit: int = 100,
    offset: int = 0,
    category: Optional[str] = None,
    sfx_library: SoundEffectsLibrary = Depends(get_sound_effects_library)
):
    """
    Get all sound effects with pagination.
    """
    try:
        result = sfx_library.get_all_sound_effects(
            limit=limit,
            offset=offset,
            category=category
        )
        return result
    except Exception as e:
        logger.error(f"Error getting all sound effects: {str(e)}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Failed to get all sound effects: {str(e)}"
        )

@router.get("/library/statistics", response_model=Dict[str, Any], tags=["sound_effects"])
async def get_statistics(
    sfx_library: SoundEffectsLibrary = Depends(get_sound_effects_library)
):
    """
    Get statistics about the sound effects library.
    """
    try:
        result = sfx_library.get_statistics()
        return result
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Failed to get statistics: {str(e)}"
        )

# API Endpoints for Sound Effects Processor
@router.post("/processor/spatial-positioning", response_model=Dict[str, Any], tags=["sound_effects"])
async def apply_spatial_positioning(
    request: SpatialPositioningRequest,
    sfx_processor: SoundEffectsProcessor = Depends(get_sound_effects_processor)
):
    """
    Apply spatial positioning to a sound effect.
    """
    try:
        result = sfx_processor.apply_spatial_positioning(
            input_file=request.input_file,
            output_file=request.output_file,
            position=request.position.dict(),
            channels=request.channels,
            intensity=request.intensity
        )
        return result
    except Exception as e:
        logger.error(f"Error applying spatial positioning: {str(e)}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Failed to apply spatial positioning: {str(e)}"
        )

@router.post("/processor/adjust-intensity", response_model=Dict[str, Any], tags=["sound_effects"])
async def adjust_intensity(
    request: IntensityAdjustmentRequest,
    sfx_processor: SoundEffectsProcessor = Depends(get_sound_effects_processor)
):
    """
    Adjust the intensity of a sound effect.
    """
    try:
        result = sfx_processor.adjust_intensity(
            input_file=request.input_file,
            output_file=request.output_file,
            intensity=request.intensity,
            fade_in=request.fade_in,
            fade_out=request.fade_out
        )
        return result
    except Exception as e:
        logger.error(f"Error adjusting intensity: {str(e)}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Failed to adjust intensity: {str(e)}"
        )

@router.post("/processor/add-to-video", response_model=Dict[str, Any], tags=["sound_effects"])
async def add_sound_effect_to_video(
    request: AddToVideoRequest,
    sfx_processor: SoundEffectsProcessor = Depends(get_sound_effects_processor)
):
    """
    Add a sound effect to a video.
    """
    try:
        result = sfx_processor.add_sound_effect_to_video(
            video_file=request.video_file,
            sound_file=request.sound_file,
            output_file=request.output_file,
            start_time=request.start_time,
            intensity=request.intensity,
            position=request.position.dict() if request.position else None,
            keep_original_audio=request.keep_original_audio
        )
        return result
    except Exception as e:
        logger.error(f"Error adding sound effect to video: {str(e)}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Failed to add sound effect to video: {str(e)}"
        )

@router.post("/processor/mix-multiple", response_model=Dict[str, Any], tags=["sound_effects"])
async def mix_multiple_sound_effects(
    request: MixMultipleRequest,
    sfx_processor: SoundEffectsProcessor = Depends(get_sound_effects_processor)
):
    """
    Mix multiple sound effects together.
    """
    try:
        sound_files = [
            {
                "file_path": sf.file_path,
                "start_time": sf.start_time,
                "intensity": sf.intensity,
                "position": sf.position.dict() if sf.position else None
            }
            for sf in request.sound_files
        ]
        
        result = sfx_processor.mix_multiple_sound_effects(
            sound_files=sound_files,
            output_file=request.output_file,
            target_duration=request.target_duration,
            master_volume=request.master_volume
        )
        return result
    except Exception as e:
        logger.error(f"Error mixing sound effects: {str(e)}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Failed to mix sound effects: {str(e)}"
        )

@router.post("/processor/apply-to-video", response_model=Dict[str, Any], tags=["sound_effects"])
async def apply_sound_effects_to_video(
    request: ApplyToVideoRequest,
    sfx_processor: SoundEffectsProcessor = Depends(get_sound_effects_processor)
):
    """
    Apply multiple sound effects to a video.
    """
    try:
        sound_effects = [
            {
                "file_path": sf.file_path,
                "start_time": sf.start_time,
                "intensity": sf.intensity,
                "position": sf.position.dict() if sf.position else None
            }
            for sf in request.sound_effects
        ]
        
        result = sfx_processor.apply_sound_effects_to_video(
            video_file=request.video_file,
            output_file=request.output_file,
            sound_effects=sound_effects,
            keep_original_audio=request.keep_original_audio,
            master_volume=request.master_volume
        )
        return result
    except Exception as e:
        logger.error(f"Error applying sound effects to video: {str(e)}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Failed to apply sound effects to video: {str(e)}"
        )

@router.post("/processor/analyze-scene", response_model=Dict[str, Any], tags=["sound_effects"])
async def analyze_scene_intensity(
    request: AnalyzeSceneRequest,
    sfx_processor: SoundEffectsProcessor = Depends(get_sound_effects_processor)
):
    """
    Analyze a video scene to determine intensity levels over time.
    
    This endpoint processes a video file to identify dynamic changes in scene intensity,
    which can be used for automatic sound effect selection and intensity adjustment.
    
    Returns intensity values for each segment of the video, based on specified segment duration.
    """
    try:
        result = sfx_processor.analyze_scene_intensity(
            video_file=request.video_file,
            segment_duration=request.segment_duration
        )
        return result
    except Exception as e:
        logger.error(f"Error analyzing scene intensity: {str(e)}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Error analyzing scene intensity: {str(e)}"
        )

@router.post("/recommend", response_model=Dict[str, Any], tags=["sound_effects"])
async def recommend_sound_effects(
    request: RecommendSoundEffectsRequest,
    sfx_library: SoundEffectsLibrary = Depends(get_sound_effects_library)
):
    """
    Provide context-aware sound effect recommendations based on video content analysis.
    
    This endpoint leverages semantic analysis of the provided context (transcript, scene descriptions,
    mood, etc.) to recommend the most appropriate sound effects for the given scene or moment.
    
    The recommendations are ranked by relevance score and consider factors such as:
    - Semantic matching with transcript and scene descriptions
    - Trigger word detection
    - Category and mood compatibility
    - Intensity appropriateness
    - Avoidance of repetition (through preceding_effects)
    """
    try:
        # Create context dictionary from request
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
        
        # Get recommendations from library
        result = sfx_library.recommend_sound_effects(
            context=context,
            max_results=request.max_results,
            include_details=request.include_details
        )
        
        return result
    except Exception as e:
        logger.error(f"Error recommending sound effects: {str(e)}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Error recommending sound effects: {str(e)}"
        ) 