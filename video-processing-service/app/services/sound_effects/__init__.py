"""
Sound Effects Module

This module provides functionality for managing and processing sound effects
including context-aware recommendations, spatial audio positioning, and 
custom library integration.
"""

from app.services.sound_effects.sound_effects_library import SoundEffectsLibrary
from app.services.sound_effects.sound_effects_processor import SoundEffectsProcessor
from app.services.sound_effects.sound_effects_recommender import SoundEffectsRecommender

__all__ = [
    "SoundEffectsLibrary",
    "SoundEffectsProcessor",
    "SoundEffectsRecommender"
] 