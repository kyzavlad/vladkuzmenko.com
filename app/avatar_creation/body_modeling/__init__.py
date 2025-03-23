"""
Body modeling module for avatar creation.
Includes classes for body mesh generation, pose estimation, measurement, 
texture mapping, and integration with face models.
"""

# Import body modeling classes
from app.avatar_creation.body_modeling.pose_estimation import PoseEstimator
from app.avatar_creation.body_modeling.body_mesh_generator import BodyMeshGenerator
from app.avatar_creation.body_modeling.body_measurement import BodyMeasurement
from app.avatar_creation.body_modeling.body_texture_mapper import BodyTextureMapper
from app.avatar_creation.body_modeling.face_body_integration import FaceBodyIntegrator

# Import utility functions
from app.avatar_creation.face_modeling.utils import (
    load_image,
    save_image,
    preprocess_image,
    tensor_to_image,
    image_to_tensor,
    get_device,
    ensure_directory
)

# Define public exports
__all__ = [
    'PoseEstimator',
    'BodyMeshGenerator',
    'BodyMeasurement',
    'BodyTextureMapper',
    'FaceBodyIntegrator',
] 