"""
Avatar Creation Module

This module provides functionality for creating 3D avatars from images or video.
It includes face modeling, body modeling, animation capabilities, and voice cloning.
"""

# Import from face modeling submodule
from app.avatar_creation.face_modeling import (
    FaceReconstructor, FaceTextureMapper, 
    FaceGeometryRefiner, ExpressionCapture
)

# Import from body modeling submodule
from app.avatar_creation.body_modeling import (
    PoseEstimator, BodyMeshGenerator, 
    BodyMeasurement, BodyTextureMapper,
    FaceBodyIntegrator
)

# Import from animation submodule
from app.avatar_creation.animation import (
    AvatarAnimator, FacialLandmarkTracker,
    MicroExpressionSynthesizer, GazeController,
    HeadPoseController, EmotionController,
    FirstOrderMotionModel, GestureMannerismLearner
)

# Import from voice cloning submodule
from app.avatar_creation.voice_cloning import (
    VoiceCloner, VoiceCharacteristicExtractor,
    SpeakerEmbedding, NeuralVocoder
)

# Import utility functions
from app.avatar_creation.face_modeling.utils import (
    load_image, save_image, preprocess_image,
    tensor_to_image, image_to_tensor,
    get_device, ensure_directory
)

# Define the public API
__all__ = [
    # Face modeling classes
    'FaceReconstructor', 
    'FaceTextureMapper',
    'FaceGeometryRefiner',
    'ExpressionCapture',
    
    # Body modeling classes
    'PoseEstimator',
    'BodyMeshGenerator',
    'BodyMeasurement',
    'BodyTextureMapper',
    'FaceBodyIntegrator',
    
    # Animation classes
    'AvatarAnimator',
    'FacialLandmarkTracker',
    'MicroExpressionSynthesizer',
    'GazeController',
    'HeadPoseController',
    'EmotionController',
    'FirstOrderMotionModel',
    'GestureMannerismLearner',
    
    # Voice cloning classes
    'VoiceCloner',
    'VoiceCharacteristicExtractor',
    'SpeakerEmbedding',
    'NeuralVocoder',
    
    # Utility functions
    'load_image',
    'save_image',
    'preprocess_image',
    'tensor_to_image',
    'image_to_tensor',
    'get_device',
    'ensure_directory'
]
