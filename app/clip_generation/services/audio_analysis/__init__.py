"""
Audio Analysis Package for Clip Generation Microservice

This package provides advanced audio analysis capabilities for the
Clip Generation Microservice, including silence detection, voice activity
detection, spectral analysis, and filler sound detection.
"""

from app.clip_generation.services.audio_analysis.audio_analyzer import AudioSegment, AudioAnalysisConfig
from app.clip_generation.services.audio_analysis.vad import VADProcessor
from app.clip_generation.services.audio_analysis.spectral_analyzer import SpectralAnalyzer, SoundType
from app.clip_generation.services.audio_analysis.filler_detector import FillerWordDetector
from app.clip_generation.services.audio_analysis.silence_detector import SilenceDetector, SilenceDetectorConfig
from app.clip_generation.services.audio_analysis.silence_processor import SilenceProcessor, SilenceProcessorConfig

__all__ = [
    'AudioSegment',
    'AudioAnalysisConfig',
    'VADProcessor',
    'SpectralAnalyzer',
    'SoundType',
    'FillerWordDetector',
    'SilenceDetector',
    'SilenceDetectorConfig',
    'SilenceProcessor',
    'SilenceProcessorConfig',
]

__version__ = '1.0.0' 