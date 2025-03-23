"""
Core modules for the AI Video Processing Platform.

This package contains the main functionality for the platform, including:
- Transcription using Whisper API
- Pause detection and removal
- Subtitle generation
- Video enhancement
- B-roll suggestions
- Music recommendations
- Sound effect suggestions
"""

from flask import current_app

__all__ = [
    'transcription',
    'pause_detection',
    'subtitle_generator',
    'video_enhancement',
    'b_roll_suggestions',
    'music_recommendation',
    'sound_effects'
] 