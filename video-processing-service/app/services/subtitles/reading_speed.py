import logging
import re
import math
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

class AudienceType(str, Enum):
    """Target audience types for reading speed configurations."""
    CHILDREN = "children"           # Young viewers, language learners (slow)
    GENERAL = "general"             # General audience (medium)
    EXPERIENCED = "experienced"     # Regular subtitle users (fast)
    SPEED_READER = "speed_reader"   # Very fast readers


class ReadingSpeedCalculator:
    """
    Calculates optimal subtitle duration based on text content and reading speed.
    
    Features:
    - Multiple calculation methods (character-based, word-based, syllable-based)
    - Audience-specific presets
    - Language-specific adjustments
    - Minimum and maximum duration constraints
    - Special content handling (numbers, technical terms)
    """
    
    # Default words per minute for different audience types
    DEFAULT_WPM = {
        AudienceType.CHILDREN: 120,      # 120 words per minute
        AudienceType.GENERAL: 160,       # 160 words per minute
        AudienceType.EXPERIENCED: 200,   # 200 words per minute
        AudienceType.SPEED_READER: 240,  # 240 words per minute
    }
    
    # Default characters per minute for different audience types
    DEFAULT_CPM = {
        AudienceType.CHILDREN: 500,      # 500 characters per minute
        AudienceType.GENERAL: 700,       # 700 characters per minute
        AudienceType.EXPERIENCED: 900,   # 900 characters per minute
        AudienceType.SPEED_READER: 1100, # 1100 characters per minute
    }
    
    # Default syllables per minute for different audience types
    DEFAULT_SPM = {
        AudienceType.CHILDREN: 200,      # 200 syllables per minute
        AudienceType.GENERAL: 280,       # 280 syllables per minute
        AudienceType.EXPERIENCED: 350,   # 350 syllables per minute
        AudienceType.SPEED_READER: 420,  # 420 syllables per minute
    }
    
    # Language-specific reading speed adjustments (multiplier)
    LANGUAGE_ADJUSTMENTS = {
        "en": 1.0,    # English (baseline)
        "fr": 0.9,    # French (slower due to more syllables per word)
        "es": 0.95,   # Spanish
        "de": 0.85,   # German (compound words)
        "it": 0.95,   # Italian
        "pt": 0.95,   # Portuguese
        "nl": 0.9,    # Dutch
        "ru": 0.85,   # Russian (Cyrillic script)
        "ja": 0.7,    # Japanese (complex writing system)
        "zh": 0.75,   # Chinese (character-based)
        "ko": 0.8,    # Korean
        "ar": 0.8,    # Arabic (right-to-left)
        "hi": 0.85,   # Hindi
        "th": 0.8,    # Thai
    }
    
    # Minimum and maximum duration constraints (in seconds)
    MIN_DURATION = 0.7   # Minimum subtitle duration
    MAX_DURATION = 7.0   # Maximum subtitle duration
    
    # Special content patterns that might require more reading time
    SPECIAL_CONTENT = {
        r'\d+\.\d+': 1.2,      # Decimal numbers (e.g. 3.14159)
        r'[A-Z]{2,}': 1.3,     # ALL CAPS words
        r'[\w\.-]+@[\w\.-]+': 1.5,  # Email addresses
        r'https?://[\w\./]+': 1.5,  # URLs
        r'\d{4,}': 1.3,        # Numbers with 4+ digits
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the reading speed calculator with configuration options.
        
        Args:
            config: Configuration dictionary with options:
                - audience_type: Target audience type (default: 'general')
                - calculation_method: 'word', 'character', or 'syllable' (default: 'character')
                - custom_wpm: Custom words per minute setting (overrides preset)
                - custom_cpm: Custom characters per minute setting
                - custom_spm: Custom syllables per minute setting
                - min_duration: Minimum subtitle duration in seconds
                - max_duration: Maximum subtitle duration in seconds
                - language: ISO language code for language-specific adjustments
                - adjust_for_complexity: Whether to adjust for text complexity
                - special_content_adjustment: Whether to adjust for special content
        """
        self.config = config or {}
        
        # Set audience type and calculation method
        self.audience_type = self.config.get('audience_type', AudienceType.GENERAL)
        self.calculation_method = self.config.get('calculation_method', 'character')
        
        # Set language and adjustment factor
        self.language = self.config.get('language', 'en')
        self.language_factor = self.LANGUAGE_ADJUSTMENTS.get(self.language, 1.0)
        
        # Set reading speed values
        self.wpm = self.config.get('custom_wpm', self.DEFAULT_WPM[self.audience_type])
        self.cpm = self.config.get('custom_cpm', self.DEFAULT_CPM[self.audience_type])
        self.spm = self.config.get('custom_spm', self.DEFAULT_SPM[self.audience_type])
        
        # Set duration constraints
        self.min_duration = self.config.get('min_duration', self.MIN_DURATION)
        self.max_duration = self.config.get('max_duration', self.MAX_DURATION)
        
        # Set adjustment flags
        self.adjust_for_complexity = self.config.get('adjust_for_complexity', True)
        self.special_content_adjustment = self.config.get('special_content_adjustment', True)
        
        # Initialize syllable counting for relevant languages
        self.syllable_counter = self._initialize_syllable_counter()
    
    def _initialize_syllable_counter(self):
        """Initialize syllable counter for the current language if possible."""
        # For now, we only support English syllable counting
        # More languages can be added in the future
        if self.language == 'en':
            try:
                import pyphen
                return pyphen.Pyphen(lang='en_US')
            except ImportError:
                logger.warning("Pyphen not available for syllable counting, falling back to estimation")
        return None
    
    def calculate_duration(self, text: str) -> float:
        """
        Calculate the optimal duration for a subtitle based on its text content.
        
        Args:
            text: The subtitle text
            
        Returns:
            Recommended duration in seconds
        """
        if not text:
            return self.min_duration
        
        # Choose calculation method
        if self.calculation_method == 'word':
            base_duration = self._calculate_by_words(text)
        elif self.calculation_method == 'syllable':
            base_duration = self._calculate_by_syllables(text)
        else:  # Default to character-based
            base_duration = self._calculate_by_characters(text)
        
        # Apply language adjustment
        duration = base_duration / self.language_factor
        
        # Apply complexity adjustment if enabled
        if self.adjust_for_complexity:
            complexity_factor = self._analyze_complexity(text)
            duration *= complexity_factor
        
        # Apply special content adjustment if enabled
        if self.special_content_adjustment:
            special_factor = self._check_special_content(text)
            duration *= special_factor
        
        # Ensure duration is within constraints
        return max(min(duration, self.max_duration), self.min_duration)
    
    def _calculate_by_words(self, text: str) -> float:
        """Calculate duration based on word count and words per minute."""
        # Count words (simple split by spaces)
        word_count = len(text.split())
        
        # Calculate base duration in seconds
        return (word_count / self.wpm) * 60
    
    def _calculate_by_characters(self, text: str) -> float:
        """Calculate duration based on character count and characters per minute."""
        # Count characters (excluding whitespace)
        char_count = len(re.sub(r'\s', '', text))
        
        # Calculate base duration in seconds
        return (char_count / self.cpm) * 60
    
    def _calculate_by_syllables(self, text: str) -> float:
        """Calculate duration based on syllable count and syllables per minute."""
        syllable_count = self._count_syllables(text)
        
        # Calculate base duration in seconds
        return (syllable_count / self.spm) * 60
    
    def _count_syllables(self, text: str) -> int:
        """
        Count syllables in text.
        
        Uses pyphen library if available, otherwise uses a simple estimation.
        """
        if self.syllable_counter:
            # Use pyphen for more accurate syllable counting
            words = text.split()
            syllable_count = sum(len(self.syllable_counter.positions(word)) + 1 for word in words)
            return syllable_count
        else:
            # Simple estimation method for English
            if self.language == 'en':
                return self._estimate_english_syllables(text)
            
            # For other languages, fall back to character count / 3
            # This is a very rough approximation
            return len(re.sub(r'\s', '', text)) // 3
    
    def _estimate_english_syllables(self, text: str) -> int:
        """Simple estimation of syllables for English text."""
        text = text.lower()
        text = re.sub(r'[^a-z]', ' ', text)
        words = text.split()
        
        count = 0
        for word in words:
            word_count = 0
            
            # Count vowel groups
            vowel_groups = re.finditer(r'[aeiouy]+', word)
            for _ in vowel_groups:
                word_count += 1
            
            # Adjust for silent e at the end
            if word.endswith('e') and len(word) > 2 and word[-2] not in 'aeiouy':
                word_count -= 1
            
            # Ensure at least one syllable per word
            word_count = max(word_count, 1)
            count += word_count
        
        return count
    
    def _analyze_complexity(self, text: str) -> float:
        """Analyze text complexity and return an adjustment factor."""
        # Simple complexity analysis based on average word length
        words = text.split()
        if not words:
            return 1.0
        
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Longer words generally indicate more complex text
        if avg_word_length > 6.5:
            return 1.3  # Very complex
        elif avg_word_length > 5.5:
            return 1.2  # Complex
        elif avg_word_length > 4.5:
            return 1.1  # Moderately complex
        else:
            return 1.0  # Simple
    
    def _check_special_content(self, text: str) -> float:
        """Check for special content that might require more reading time."""
        max_factor = 1.0
        
        for pattern, factor in self.SPECIAL_CONTENT.items():
            if re.search(pattern, text):
                max_factor = max(max_factor, factor)
        
        return max_factor
    
    def calibrate_subtitle_durations(self, subtitles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calibrate the durations of a list of subtitles based on their text content.
        
        Args:
            subtitles: List of subtitle dictionaries with at least 'text', 'start', and 'end' keys
            
        Returns:
            List of subtitles with optimized durations
        """
        result = []
        
        for i, subtitle in enumerate(subtitles):
            # Calculate optimal duration for this subtitle
            optimal_duration = self.calculate_duration(subtitle['text'])
            
            # Calculate new end time
            new_end = subtitle['start'] + optimal_duration
            
            # Check for collision with next subtitle
            if i < len(subtitles) - 1 and new_end > subtitles[i + 1]['start']:
                # Adjust to avoid collision, with a small gap
                new_end = subtitles[i + 1]['start'] - 0.05
            
            # Ensure minimum duration
            if new_end - subtitle['start'] < self.min_duration:
                new_end = subtitle['start'] + self.min_duration
            
            # Create adjusted subtitle
            adjusted_subtitle = subtitle.copy()
            adjusted_subtitle['end'] = new_end
            adjusted_subtitle['duration'] = new_end - subtitle['start']
            
            result.append(adjusted_subtitle)
        
        return result
    
    def set_audience_type(self, audience_type: AudienceType) -> None:
        """
        Change the target audience type and update reading speed settings.
        
        Args:
            audience_type: New target audience type
        """
        self.audience_type = audience_type
        
        # Update reading speeds unless custom values were provided
        if 'custom_wpm' not in self.config:
            self.wpm = self.DEFAULT_WPM[audience_type]
        
        if 'custom_cpm' not in self.config:
            self.cpm = self.DEFAULT_CPM[audience_type]
            
        if 'custom_spm' not in self.config:
            self.spm = self.DEFAULT_SPM[audience_type]
    
    def get_reading_speeds(self) -> Dict[str, int]:
        """
        Get the current reading speed settings.
        
        Returns:
            Dictionary with words per minute, characters per minute, and syllables per minute
        """
        return {
            'wpm': self.wpm,
            'cpm': self.cpm,
            'spm': self.spm,
            'audience_type': self.audience_type,
            'language': self.language,
            'language_factor': self.language_factor
        } 