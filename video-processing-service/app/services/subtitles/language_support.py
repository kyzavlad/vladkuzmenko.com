import logging
import re
import unicodedata
from enum import Enum
from typing import Dict, Any, Optional, List, Set, Tuple
import langdetect

logger = logging.getLogger(__name__)

class TextDirection(str, Enum):
    """Text direction for different writing systems."""
    LTR = "ltr"  # Left-to-right (Latin, Cyrillic, etc.)
    RTL = "rtl"  # Right-to-left (Arabic, Hebrew, etc.)
    TTB = "ttb"  # Top-to-bottom (Traditional Chinese, Japanese, etc.)


class LanguageScript(str, Enum):
    """Major writing scripts for language support."""
    LATIN = "latin"          # Latin script (English, Spanish, French, etc.)
    CYRILLIC = "cyrillic"    # Cyrillic script (Russian, Ukrainian, etc.)
    ARABIC = "arabic"        # Arabic script
    HEBREW = "hebrew"        # Hebrew script
    CJK = "cjk"              # Chinese, Japanese, Korean
    DEVANAGARI = "devanagari"  # Hindi, Sanskrit, etc.
    THAI = "thai"            # Thai script
    KOREAN = "korean"        # Korean Hangul
    GREEK = "greek"          # Greek script
    OTHER = "other"          # Other scripts


class LanguageSupport:
    """
    Provides language support and character rendering for the subtitle system.
    
    Features:
    - Language detection for automatic configuration
    - Script identification and character set handling
    - Font selection for different scripts
    - Text direction handling (LTR, RTL, TTB)
    - Unicode normalization for consistent display
    - Special character handling and fallbacks
    """
    
    # Map of language codes to their text direction
    LANGUAGE_DIRECTIONS = {
        # Right-to-left languages
        'ar': TextDirection.RTL,  # Arabic
        'fa': TextDirection.RTL,  # Persian
        'he': TextDirection.RTL,  # Hebrew
        'ur': TextDirection.RTL,  # Urdu
        
        # Top-to-bottom languages (traditional)
        # Modern CJK is usually written LTR, but some traditional formats use TTB
        'zh-traditional': TextDirection.TTB,  # Traditional Chinese
        'ja-traditional': TextDirection.TTB,  # Traditional Japanese
        
        # All other languages default to LTR
    }
    
    # Map of language codes to script
    LANGUAGE_SCRIPTS = {
        # Latin script
        'en': LanguageScript.LATIN,  # English
        'es': LanguageScript.LATIN,  # Spanish
        'fr': LanguageScript.LATIN,  # French
        'de': LanguageScript.LATIN,  # German
        'it': LanguageScript.LATIN,  # Italian
        'pt': LanguageScript.LATIN,  # Portuguese
        'nl': LanguageScript.LATIN,  # Dutch
        'sv': LanguageScript.LATIN,  # Swedish
        'no': LanguageScript.LATIN,  # Norwegian
        'da': LanguageScript.LATIN,  # Danish
        'fi': LanguageScript.LATIN,  # Finnish
        'pl': LanguageScript.LATIN,  # Polish
        'cs': LanguageScript.LATIN,  # Czech
        'hu': LanguageScript.LATIN,  # Hungarian
        'ro': LanguageScript.LATIN,  # Romanian
        'vi': LanguageScript.LATIN,  # Vietnamese
        'id': LanguageScript.LATIN,  # Indonesian
        'ms': LanguageScript.LATIN,  # Malay
        'tr': LanguageScript.LATIN,  # Turkish
        
        # Cyrillic script
        'ru': LanguageScript.CYRILLIC,  # Russian
        'uk': LanguageScript.CYRILLIC,  # Ukrainian
        'bg': LanguageScript.CYRILLIC,  # Bulgarian
        'sr': LanguageScript.CYRILLIC,  # Serbian
        'mk': LanguageScript.CYRILLIC,  # Macedonian
        
        # Arabic script
        'ar': LanguageScript.ARABIC,  # Arabic
        'fa': LanguageScript.ARABIC,  # Persian
        'ur': LanguageScript.ARABIC,  # Urdu
        
        # Hebrew script
        'he': LanguageScript.HEBREW,  # Hebrew
        'yi': LanguageScript.HEBREW,  # Yiddish
        
        # CJK (Chinese, Japanese, Korean)
        'zh': LanguageScript.CJK,  # Chinese
        'ja': LanguageScript.CJK,  # Japanese
        'ko': LanguageScript.KOREAN,  # Korean
        
        # Devanagari script
        'hi': LanguageScript.DEVANAGARI,  # Hindi
        'ne': LanguageScript.DEVANAGARI,  # Nepali
        'sa': LanguageScript.DEVANAGARI,  # Sanskrit
        
        # Thai script
        'th': LanguageScript.THAI,  # Thai
        
        # Greek script
        'el': LanguageScript.GREEK,  # Greek
    }
    
    # Recommended fonts for each script
    SCRIPT_FONTS = {
        LanguageScript.LATIN: ["Arial", "Helvetica", "Roboto", "Open Sans"],
        LanguageScript.CYRILLIC: ["Arial", "Roboto", "Noto Sans", "DejaVu Sans"],
        LanguageScript.ARABIC: ["Noto Sans Arabic", "Arial Unicode MS", "Tahoma"],
        LanguageScript.HEBREW: ["Noto Sans Hebrew", "Arial Unicode MS", "Tahoma"],
        LanguageScript.CJK: ["Noto Sans CJK", "MS Gothic", "SimSun", "WenQuanYi Zen Hei"],
        LanguageScript.DEVANAGARI: ["Noto Sans Devanagari", "Mangal", "Arial Unicode MS"],
        LanguageScript.THAI: ["Noto Sans Thai", "Tahoma", "Arial Unicode MS"],
        LanguageScript.KOREAN: ["Noto Sans KR", "Malgun Gothic", "Gulim"],
        LanguageScript.GREEK: ["Noto Sans Greek", "Arial Unicode MS", "Tahoma"],
        LanguageScript.OTHER: ["Noto Sans", "Arial Unicode MS", "DejaVu Sans"]
    }
    
    # Unicode normalization form to use for each script
    SCRIPT_NORMALIZATION = {
        LanguageScript.LATIN: 'NFC',     # Canonical decomposition, followed by canonical composition
        LanguageScript.CYRILLIC: 'NFC',
        LanguageScript.ARABIC: 'NFKC',   # Compatibility decomposition, followed by canonical composition
        LanguageScript.HEBREW: 'NFKC',
        LanguageScript.CJK: 'NFC',
        LanguageScript.DEVANAGARI: 'NFC',
        LanguageScript.THAI: 'NFC',
        LanguageScript.KOREAN: 'NFC',
        LanguageScript.GREEK: 'NFC',
        LanguageScript.OTHER: 'NFKC'
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize language support with configuration options.
        
        Args:
            config: Configuration dictionary with options:
                - default_language: Default language code to use (ISO 639-1)
                - fallback_font: Fallback font to use when recommended fonts are not available
                - auto_detect_language: Whether to auto-detect language from text
                - normalize_unicode: Whether to normalize Unicode for consistent display
                - custom_script_fonts: Custom font preferences for specific scripts
                - force_bidi_support: Force bidirectional text support even for LTR formats
                - custom_font_mappings: Custom language to font mappings
                - enable_rtl_support: Whether to enable right-to-left text support
        """
        self.config = config or {}
        
        # Set default language
        self.default_language = self.config.get('default_language', 'en')
        
        # Set fallback font
        self.fallback_font = self.config.get('fallback_font', 'Arial Unicode MS')
        
        # Whether to auto-detect language
        self.auto_detect_language = self.config.get('auto_detect_language', True)
        
        # Whether to normalize Unicode
        self.normalize_unicode = self.config.get('normalize_unicode', True)
        
        # Add custom script fonts if provided
        custom_fonts = self.config.get('custom_script_fonts', {})
        for script, fonts in custom_fonts.items():
            if script in self.SCRIPT_FONTS:
                # Prepend custom fonts to the standard list
                self.SCRIPT_FONTS[script] = fonts + self.SCRIPT_FONTS[script]
        
        # Custom font mappings
        self.custom_font_mappings = self.config.get('custom_font_mappings', {})
        
        # RTL support
        self.enable_rtl_support = self.config.get('enable_rtl_support', True)
        
        # Force bidirectional text support
        self.force_bidi_support = self.config.get('force_bidi_support', False)
        
        # Initialize language detection
        self._initialize_language_detection()
    
    def _initialize_language_detection(self):
        """Initialize language detection functionality."""
        try:
            import langdetect
            self.langdetect_available = True
        except ImportError:
            logger.warning("langdetect not installed, language auto-detection will be disabled")
            self.langdetect_available = False
            self.auto_detect_language = False
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the given text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code (ISO 639-1)
        """
        if not self.langdetect_available or not self.auto_detect_language:
            return self.default_language
        
        try:
            return langdetect.detect(text)
        except:
            logger.warning("Language detection failed, using default language")
            return self.default_language
    
    def get_script_for_language(self, language_code: str) -> LanguageScript:
        """
        Get the writing script for a language.
        
        Args:
            language_code: Language code (ISO 639-1)
            
        Returns:
            Writing script enum
        """
        return self.LANGUAGE_SCRIPTS.get(language_code, LanguageScript.OTHER)
    
    def get_text_direction(self, language_code: str) -> TextDirection:
        """
        Get the text direction for a language.
        
        Args:
            language_code: Language code (ISO 639-1)
            
        Returns:
            Text direction enum
        """
        if not self.enable_rtl_support:
            return TextDirection.LTR
            
        return self.LANGUAGE_DIRECTIONS.get(language_code, TextDirection.LTR)
    
    def get_recommended_fonts(self, language_code: str) -> List[str]:
        """
        Get recommended fonts for a language.
        
        Args:
            language_code: Language code (ISO 639-1)
            
        Returns:
            List of recommended font names
        """
        # Check for custom mapping first
        if language_code in self.custom_font_mappings:
            return self.custom_font_mappings[language_code]
        
        # Get script for language
        script = self.get_script_for_language(language_code)
        
        # Get recommended fonts for script
        fonts = self.SCRIPT_FONTS.get(script, self.SCRIPT_FONTS[LanguageScript.OTHER])
        
        # Ensure fallback font is included
        if self.fallback_font not in fonts:
            fonts.append(self.fallback_font)
        
        return fonts
    
    def get_best_font(self, language_code: str) -> str:
        """
        Get the best font for a language.
        
        Args:
            language_code: Language code (ISO 639-1)
            
        Returns:
            Font name
        """
        fonts = self.get_recommended_fonts(language_code)
        return fonts[0] if fonts else self.fallback_font
    
    def normalize_text(self, text: str, language_code: str) -> str:
        """
        Normalize text for consistent display.
        
        Args:
            text: Text to normalize
            language_code: Language code (ISO 639-1)
            
        Returns:
            Normalized text
        """
        if not self.normalize_unicode:
            return text
        
        # Get script for language
        script = self.get_script_for_language(language_code)
        
        # Get normalization form for script
        norm_form = self.SCRIPT_NORMALIZATION.get(script, 'NFC')
        
        # Normalize text
        return unicodedata.normalize(norm_form, text)
    
    def apply_language_formatting(
        self,
        text: str,
        language_code: Optional[str] = None,
        format_name: str = 'vtt'
    ) -> str:
        """
        Apply language-specific formatting to text.
        
        Args:
            text: Text to format
            language_code: Language code (ISO 639-1) or None to auto-detect
            format_name: Subtitle format name (vtt, srt, ass, etc.)
            
        Returns:
            Formatted text
        """
        # Detect language if not provided
        if not language_code and self.auto_detect_language:
            language_code = self.detect_language(text)
        elif not language_code:
            language_code = self.default_language
        
        # Normalize text
        text = self.normalize_text(text, language_code)
        
        # Get text direction
        direction = self.get_text_direction(language_code)
        
        # Apply special formatting based on subtitle format and text direction
        if direction == TextDirection.RTL:
            text = self._apply_rtl_formatting(text, format_name)
        elif direction == TextDirection.TTB:
            text = self._apply_ttb_formatting(text, format_name)
        
        return text
    
    def _apply_rtl_formatting(self, text: str, format_name: str) -> str:
        """
        Apply right-to-left text formatting.
        
        Args:
            text: Text to format
            format_name: Subtitle format name
            
        Returns:
            Formatted text
        """
        # For formats that support bidirectional text natively (e.g., VTT)
        if format_name.lower() == 'vtt':
            # Add Unicode right-to-left mark at the start and set direction
            return f'<span dir="rtl">\u200F{text}</span>'
        
        # For ASS/SSA format, we can use directional override
        elif format_name.lower() in ['ass', 'ssa']:
            return f'{{\\frz180}}{text}'
        
        # For other formats, add Unicode control characters
        else:
            # Right-to-Left override character
            # Surround the text with RLO and PDF (Pop Directional Format)
            return f'\u202E{text}\u202C'
    
    def _apply_ttb_formatting(self, text: str, format_name: str) -> str:
        """
        Apply top-to-bottom text formatting.
        
        Args:
            text: Text to format
            format_name: Subtitle format name
            
        Returns:
            Formatted text
        """
        # TTB is primarily supported in ASS/SSA
        if format_name.lower() in ['ass', 'ssa']:
            # Use vertical style
            return f'{{\\an8\\fry90}}{text}'
        
        # For other formats, we don't have good TTB support
        # Just return the text as is
        return text
    
    def get_font_family_list(self, language_code: str) -> str:
        """
        Get font family list CSS string for the language.
        
        Args:
            language_code: Language code (ISO 639-1)
            
        Returns:
            CSS font-family string
        """
        fonts = self.get_recommended_fonts(language_code)
        return ', '.join(f'"{font}"' for font in fonts)
    
    def get_language_style_attributes(self, language_code: str) -> Dict[str, str]:
        """
        Get CSS style attributes for a language.
        
        Args:
            language_code: Language code (ISO 639-1)
            
        Returns:
            Dictionary of CSS style attributes
        """
        attributes = {}
        
        # Font family
        fonts = self.get_recommended_fonts(language_code)
        attributes['font-family'] = ', '.join(f'"{font}"' for font in fonts)
        
        # Text direction
        direction = self.get_text_direction(language_code)
        if direction == TextDirection.RTL:
            attributes['direction'] = 'rtl'
            attributes['unicode-bidi'] = 'bidi-override'
        elif direction == TextDirection.TTB:
            attributes['writing-mode'] = 'vertical-rl'
        
        return attributes
    
    def supports_language(self, language_code: str) -> bool:
        """
        Check if a language is fully supported.
        
        Args:
            language_code: Language code (ISO 639-1)
            
        Returns:
            True if language is fully supported
        """
        return language_code in self.LANGUAGE_SCRIPTS
    
    def format_subtitle_for_language(
        self,
        subtitle_text: str,
        language_code: Optional[str] = None,
        subtitle_format: str = 'vtt'
    ) -> str:
        """
        Format a complete subtitle for a specific language.
        
        This applies all necessary language-specific formatting:
        - Character normalization
        - Text direction
        - Font specifications (where applicable)
        
        Args:
            subtitle_text: Original subtitle text
            language_code: Language code or None to auto-detect
            subtitle_format: Subtitle format name
            
        Returns:
            Formatted subtitle text
        """
        # Detect language if not provided
        if not language_code and self.auto_detect_language:
            language_code = self.detect_language(subtitle_text)
        elif not language_code:
            language_code = self.default_language
        
        # Normalize text
        normalized_text = self.normalize_text(subtitle_text, language_code)
        
        # Apply format-specific language formatting
        if subtitle_format.lower() == 'vtt':
            return self._format_vtt_for_language(normalized_text, language_code)
        elif subtitle_format.lower() in ['ass', 'ssa']:
            return self._format_ass_for_language(normalized_text, language_code)
        elif subtitle_format.lower() == 'ttml':
            return self._format_ttml_for_language(normalized_text, language_code)
        else:
            # For formats that don't support language-specific formatting,
            # just apply basic text direction if needed
            direction = self.get_text_direction(language_code)
            if direction == TextDirection.RTL:
                return f'\u202E{normalized_text}\u202C'  # RLO and PDF
            return normalized_text
    
    def _format_vtt_for_language(self, text: str, language_code: str) -> str:
        """Format WebVTT subtitle for specific language."""
        direction = self.get_text_direction(language_code)
        font_family = self.get_font_family_list(language_code)
        
        if direction == TextDirection.RTL:
            return f'<span dir="rtl" style="font-family: {font_family};">\u200F{text}</span>'
        elif direction == TextDirection.TTB:
            # WebVTT doesn't properly support vertical text, so just use RTL as approximation
            return f'<span style="font-family: {font_family};">{text}</span>'
        else:
            return f'<span style="font-family: {font_family};">{text}</span>'
    
    def _format_ass_for_language(self, text: str, language_code: str) -> str:
        """Format ASS/SSA subtitle for specific language."""
        direction = self.get_text_direction(language_code)
        font_name = self.get_best_font(language_code)
        
        if direction == TextDirection.RTL:
            return f'{{\\fnname({font_name})\\frz180}}{text}'
        elif direction == TextDirection.TTB:
            return f'{{\\fnname({font_name})\\fry90}}{text}'
        else:
            return f'{{\\fnname({font_name})}}{text}'
    
    def _format_ttml_for_language(self, text: str, language_code: str) -> str:
        """Format TTML subtitle for specific language."""
        direction = self.get_text_direction(language_code)
        font_family = self.get_font_family_list(language_code)
        
        if direction == TextDirection.RTL:
            return f'<span tts:direction="rtl" tts:fontFamily="{font_family}">{text}</span>'
        elif direction == TextDirection.TTB:
            return f'<span tts:writingMode="tb" tts:fontFamily="{font_family}">{text}</span>'
        else:
            return f'<span tts:fontFamily="{font_family}">{text}</span>'
    
    def get_language_name(self, language_code: str) -> str:
        """
        Get the display name of a language.
        
        Args:
            language_code: Language code (ISO 639-1)
            
        Returns:
            Language name in English
        """
        language_names = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ja': 'Japanese',
            'zh': 'Chinese',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'ko': 'Korean',
            'th': 'Thai',
            'vi': 'Vietnamese',
            'nl': 'Dutch',
            'tr': 'Turkish',
            'pl': 'Polish',
            'uk': 'Ukrainian',
            'fa': 'Persian',
            'he': 'Hebrew',
            'id': 'Indonesian',
            'sv': 'Swedish',
            'el': 'Greek',
            'ro': 'Romanian',
            'cs': 'Czech',
            'hu': 'Hungarian',
            'fi': 'Finnish',
            'no': 'Norwegian',
            'da': 'Danish',
            'bg': 'Bulgarian',
            'sk': 'Slovak',
            'ur': 'Urdu',
            'hr': 'Croatian',
            'sr': 'Serbian',
            'lt': 'Lithuanian',
            'lv': 'Latvian',
            'et': 'Estonian',
            'sl': 'Slovenian',
        }
        
        return language_names.get(language_code, f"Unknown ({language_code})") 