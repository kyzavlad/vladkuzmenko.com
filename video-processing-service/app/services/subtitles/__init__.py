"""
Subtitle Generation System for the Video Processing Service.

This package provides a comprehensive system for generating, rendering, and editing subtitles
for video content. It includes the following components:

- SubtitleGenerator: Generates subtitle files in various formats (SRT, VTT, ASS, etc.)
- SubtitleRenderer: Renders videos with burned-in subtitles
- SubtitleEditor: Provides editing capabilities for subtitle timing and text
- SubtitleService: Main interface that integrates all components
- SmartTextBreaker: Intelligent text breaking for subtitles with multiple strategies and language support
- SubtitlePositioningService: Analyzes video content to determine optimal subtitle positioning
- SubtitlePositioningLite: Lightweight version of positioning service for basic needs
- ReadingSpeedCalculator: Calculates optimal subtitle duration based on text content and reading speed
- EmphasisDetector: Detects and applies emphasis formatting (bold/italic) to subtitle text
- LanguageSupport: Provides multi-language support with proper character rendering for different writing systems
"""

from .subtitle_generator import (
    SubtitleGenerator, SubtitleStyle, SubtitleFormat,
    TextAlignment, TextPosition
)
from .subtitle_renderer import SubtitleRenderer, RenderQuality
from .subtitle_editor import SubtitleEditor, EditOperation
from .subtitle_service import SubtitleService
from .smart_text_breaker import SmartTextBreaker
from .subtitle_positioning import SubtitlePositioningService
from .subtitle_positioning_lite import SubtitlePositioningLite
from .reading_speed import ReadingSpeedCalculator, AudienceType
from .emphasis_detection import EmphasisDetector, EmphasisFormat
from .language_support import LanguageSupport, TextDirection, LanguageScript

__all__ = [
    'SubtitleGenerator',
    'SubtitleStyle',
    'SubtitleFormat',
    'TextAlignment',
    'TextPosition',
    'SubtitleRenderer',
    'RenderQuality',
    'SubtitleEditor',
    'EditOperation',
    'SubtitleService',
    'SmartTextBreaker',
    'SubtitlePositioningService',
    'SubtitlePositioningLite',
    'ReadingSpeedCalculator',
    'AudienceType',
    'EmphasisDetector',
    'EmphasisFormat',
    'LanguageSupport',
    'TextDirection',
    'LanguageScript',
] 