#!/usr/bin/env python3
"""
Avatar Creation - Video Translation Module

This module provides comprehensive video translation capabilities for avatar creation,
including multi-language processing, context-aware neural translation, terminology 
preservation, cultural adaptation, and script timing preservation for video dubbing.

The module supports 50+ languages with specialized handling for technical vocabulary,
maintaining timing across translations, and preserving named entities.
"""

# Import from translation submodules
from app.avatar_creation.video_translation.translator import (
    NeuralTranslator,
    ContextAwareTranslator,
    TranslationOptions
)

from app.avatar_creation.video_translation.language_manager import (
    LanguageManager,
    LanguageProfile,
    DialectVariant
)

from app.avatar_creation.video_translation.terminology import (
    TerminologyManager,
    IndustryTerminology,
    NamedEntityRecognizer,
    TechnicalVocabulary,
    TermDefinition
)

from app.avatar_creation.video_translation.cultural_adaptation import (
    CulturalAdapter,
    IdiomaticExpressionHandler,
    CulturalReference
)

from app.avatar_creation.video_translation.timing import (
    ScriptTimingPreserver,
    TimedSegment,
    LanguageTimingProfile
)

# Define the public API
__all__ = [
    # Core translator
    'NeuralTranslator',
    'ContextAwareTranslator',
    'TranslationOptions',
    
    # Language management
    'LanguageManager',
    'LanguageProfile',
    'DialectVariant',
    
    # Terminology handling
    'TerminologyManager',
    'IndustryTerminology',
    'NamedEntityRecognizer',
    'TechnicalVocabulary',
    'TermDefinition',
    
    # Cultural adaptation
    'CulturalAdapter',
    'IdiomaticExpressionHandler',
    'CulturalReference',
    
    # Timing preservation
    'ScriptTimingPreserver',
    'TimedSegment',
    'LanguageTimingProfile'
] 