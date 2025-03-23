"""
Avatar Generation Module

This package provides functionality for generating AI avatars from input images and videos,
with a focus on high-quality 3D face reconstruction and realistic animations.
"""

from .face_modeling import FaceModeling
from .animation_framework import AnimationFramework

__all__ = ["FaceModeling", "AnimationFramework"] 