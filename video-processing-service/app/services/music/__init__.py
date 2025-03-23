"""
Music Selection & Integration Module

This module provides functionality for analyzing video content and selecting
appropriate music tracks based on mood, tempo, genre, and emotional arcs.
It includes tools for BPM detection, genre classification, dynamic volume
adjustment, and integration with copyright-free music libraries.
"""

from app.services.music.music_selector import MusicSelector
from app.services.music.mood_analyzer import MoodAnalyzer
from app.services.music.bpm_detector import BPMDetector
from app.services.music.genre_classifier import GenreClassifier
from app.services.music.music_library import MusicLibrary
from app.services.music.audio_fingerprinter import AudioFingerprinter
from app.services.music.volume_adjuster import VolumeAdjuster
from app.services.music.emotional_arc_mapper import EmotionalArcMapper
from app.services.music.external_music_service import ExternalMusicService
from app.services.music.music_recommender import MusicRecommender

__all__ = [
    "MusicSelector",
    "MoodAnalyzer",
    "BPMDetector",
    "GenreClassifier",
    "MusicLibrary",
    "AudioFingerprinter",
    "VolumeAdjuster",
    "EmotionalArcMapper",
    "ExternalMusicService",
    "MusicRecommender"
] 