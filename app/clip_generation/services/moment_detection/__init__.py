"""
Interesting Moment Detection

This package provides components for detecting interesting moments in videos
through multi-faceted content analysis, engagement prediction, and narrative cohesion analysis.
"""

__version__ = '1.0.0'

# Core components
from app.clip_generation.services.moment_detection.moment_analyzer import MomentAnalyzer, MomentAnalyzerConfig, MomentType, MomentScore
from app.clip_generation.services.moment_detection.content_analysis import ContentAnalyzer, ContentAnalysisConfig, AudioAnalyzer
from app.clip_generation.services.moment_detection.voice_analysis import VoiceAnalyzer
from app.clip_generation.services.moment_detection.transcript_analysis import TranscriptAnalyzer, SentimentAnalyzer, KeywordAnalyzer

__all__ = [
    # Core analyzer classes
    'MomentAnalyzer',
    'ContentAnalyzer',
    'VoiceAnalyzer',
    'TranscriptAnalyzer',
    'SentimentAnalyzer',
    'KeywordAnalyzer',
    'AudioAnalyzer',
    
    # Configuration classes
    'MomentAnalyzerConfig',
    'ContentAnalysisConfig',
    
    # Data classes
    'MomentType',
    'MomentScore',
] 