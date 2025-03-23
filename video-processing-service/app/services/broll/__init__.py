"""
B-Roll Insertion Engine for intelligent video content enhancement.

This module provides functionality for analyzing video content,
suggesting appropriate b-roll footage, and seamlessly inserting
it into videos based on speech content and visual themes.
"""

from app.services.broll.broll_engine import BRollEngine
from app.services.broll.content_analyzer import ContentAnalyzer
from app.services.broll.scene_detector import SceneDetector
from app.services.broll.stock_integration import StockFootageProvider
from app.services.broll.semantic_matcher import SemanticMatcher

__all__ = [
    'BRollEngine',
    'ContentAnalyzer',
    'SceneDetector',
    'StockFootageProvider',
    'SemanticMatcher'
] 