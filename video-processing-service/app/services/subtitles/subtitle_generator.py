import logging
import asyncio
import os
import json
import tempfile
import subprocess
import re
import nltk
import math
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
import aiofiles

from .smart_text_breaker import SmartTextBreaker
from .emphasis_detection import EmphasisDetector, EmphasisFormat
from .language_support import LanguageSupport, TextDirection, LanguageScript


class SubtitleFormat(str, Enum):
    """Supported subtitle file formats."""
    SRT = "srt"       # SubRip Text - most widely supported
    VTT = "vtt"       # Web Video Text Tracks - for HTML5 video
    ASS = "ass"       # Advanced SubStation Alpha - supports advanced styling
    SSA = "ssa"       # SubStation Alpha - older version of ASS
    SBV = "sbv"       # YouTube's SubViewer format
    TTML = "ttml"     # Timed Text Markup Language - XML-based


class TextAlignment(str, Enum):
    """Text alignment options for subtitles."""
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class TextPosition(str, Enum):
    """Vertical position options for subtitles."""
    TOP = "top"
    MIDDLE = "middle"
    BOTTOM = "bottom"


class ReadingSpeedPreset(str, Enum):
    """Preset reading speed profiles for different audiences."""
    SLOW = "slow"               # For children, language learners (120-150 WPM)
    STANDARD = "standard"       # For general audience (160-180 WPM)
    FAST = "fast"               # For experienced viewers (200-220 WPM)
    VERY_FAST = "very_fast"     # For speed readers (240+ WPM)


@dataclass
class FormattedText:
    """Represents text with formatting information."""
    text: str
    bold: bool = False
    italic: bool = False
    underline: bool = False
    color: Optional[str] = None


@dataclass
class SubtitleStyle:
    """Style configuration for subtitles."""
    name: str
    font_family: str
    font_size: int               # Font size in pixels
    font_color: str              # Hex color code
    background_color: str        # Hex color code with alpha (e.g. "#00000080")
    bold: bool = False
    italic: bool = False
    alignment: TextAlignment = TextAlignment.CENTER
    position: TextPosition = TextPosition.BOTTOM
    outline_width: int = 0       # Outline width in pixels
    outline_color: str = "#000000"
    shadow_offset: int = 0       # Shadow offset in pixels
    shadow_color: str = "#000000"
    line_spacing: float = 1.0    # Line spacing multiplier
    max_lines: int = 2           # Maximum lines per subtitle
    max_chars_per_line: int = 42 # Maximum characters per line
    
    # Advanced styling options for formats that support them (like ASS)
    custom_style_options: Dict[str, Any] = field(default_factory=dict)
    
    # Formatting for emphasized text
    emphasis_bold: bool = True   # Whether to use bold for emphasized text
    emphasis_italic: bool = False # Whether to use italic for emphasized text
    emphasis_color: Optional[str] = None # Color for emphasized text (if applicable)
    
    def to_ass_style(self) -> str:
        """Convert to ASS style format string."""
        # Base style format: Name, FontName, FontSize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, ...
        bold_value = "-1" if self.bold else "0"
        italic_value = "-1" if self.italic else "0"
        
        # Convert hex colors to ASS format (&HAABBGGRR)
        primary_color = self._hex_to_ass_color(self.font_color)
        outline_color = self._hex_to_ass_color(self.outline_color)
        shadow_color = self._hex_to_ass_color(self.shadow_color)
        back_color = self._hex_to_ass_color(self.background_color)
        
        # Determine alignment value (1-9)
        # ASS uses a numpad-based system:
        # 7 8 9
        # 4 5 6
        # 1 2 3
        alignment_map = {
            (TextAlignment.LEFT, TextPosition.TOP): 7,
            (TextAlignment.CENTER, TextPosition.TOP): 8,
            (TextAlignment.RIGHT, TextPosition.TOP): 9,
            (TextAlignment.LEFT, TextPosition.MIDDLE): 4,
            (TextAlignment.CENTER, TextPosition.MIDDLE): 5,
            (TextAlignment.RIGHT, TextPosition.MIDDLE): 6,
            (TextAlignment.LEFT, TextPosition.BOTTOM): 1,
            (TextAlignment.CENTER, TextPosition.BOTTOM): 2,
            (TextAlignment.RIGHT, TextPosition.BOTTOM): 3,
        }
        alignment_value = alignment_map.get((self.alignment, self.position), 2)
        
        return (
            f"Style: {self.name},{self.font_family},{self.font_size},{primary_color},"
            f"{primary_color},{outline_color},{back_color},{bold_value},{italic_value},"
            f"0,0,100,100,0,0,0,0,{alignment_value},10,10,10,0"
        )
    
    def _hex_to_ass_color(self, hex_color: str) -> str:
        """Convert hex color to ASS color format.
        
        Args:
            hex_color: Hex color code (#RRGGBB or #RRGGBBAA)
            
        Returns:
            ASS color string (&HAABBGGRR)
        """
        # Remove # and handle both RGB and RGBA
        hex_color = hex_color.lstrip('#')
        
        if len(hex_color) == 6:
            # RGB format, add full opacity (00)
            hex_color = hex_color + "00"
        elif len(hex_color) == 8:
            # RGBA format, keep as is
            pass
        else:
            # Invalid format, use black with full opacity
            hex_color = "00000000"
        
        # Extract components and reorder for ASS format
        r = hex_color[0:2]
        g = hex_color[2:4]
        b = hex_color[4:6]
        a = hex_color[6:8]
        
        # Return in ASS format
        return f"&H{a}{b}{g}{r}"


class SubtitleGenerator:
    """
    Generates subtitle files in various formats with customizable styling.
    
    Features:
    - Multiple subtitle format support (SRT, VTT, ASS, etc.)
    - Customizable style templates
    - Split long lines intelligently
    - Adjust timing for better readability
    - Duration calibration based on reading speed
    - Emphasis detection for text formatting
    - Burned-in subtitle option for direct rendering
    - Smart text breaking for improved readability
    """
    
    def __init__(
        self,
        default_format: SubtitleFormat = SubtitleFormat.SRT,
        default_style: Optional[str] = "default",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the subtitle generator.
        
        Args:
            default_format: Default subtitle format
            default_style: Default style template to use
            config: Additional configuration options
        """
        self.default_format = default_format
        self.default_style = default_style
        self.config = config or {}
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize style templates
        self.style_templates = self._initialize_style_templates()
        
        # Initialize NLTK for advanced text processing
        self._initialize_nltk()
        
        # Configure reading speed settings
        self._initialize_reading_speed_settings()
        
        # Configure emphasis detection
        self._initialize_emphasis_detection()
        
        # Initialize the smart text breaker with our new implementation
        self.text_breaker = SmartTextBreaker(
            config={
                "breaking_strategy": self.config.get("breaking_strategy", "balanced"),
                "language": self.config.get("language", "en"),
                "prefer_balanced_lines": self.config.get("balance_lines", True),
                "respect_sentences": self.config.get("respect_sentences", True),
                "min_chars_per_line": self.config.get("min_line_length", 15),
                "ideal_chars_per_line": self.config.get("optimal_line_length", 42),
                "reading_speed": self.config.get("reading_speed", 20)
            }
        )
        
        # Initialize EmphasisDetector for automatic emphasis detection
        self.emphasis_detector = EmphasisDetector(config=self.config.get("emphasis_config"))
        
        # Initialize LanguageSupport for multi-language capabilities
        self.language_support = LanguageSupport(config=self.config.get("language_config"))
        
        # Default language to use (can be overridden per subtitle)
        self.default_language = self.config.get("default_language", "en")
        
        # Whether to enable automatic language detection
        self.auto_detect_language = self.config.get("auto_detect_language", True)
    
    def _initialize_nltk(self):
        """Initialize NLTK for advanced text processing."""
        self.nltk_available = False
        try:
            # Check if NLTK is available
            import nltk
            self.nltk_available = True
            
            # Download necessary NLTK resources if not already available
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
        except Exception as e:
            self.logger.warning(f"NLTK initialization failed, using basic text splitting: {str(e)}")
            self.nltk_available = False
        
        # The following lines are moved to __init__, so we remove them here
        # Configure reading speed settings
        # self._initialize_reading_speed_settings()
        
        # Configure emphasis detection
        # self._initialize_emphasis_detection()
        
        # Initialize the smart text breaker with our new implementation
        # self.text_breaker = SmartTextBreaker(
        #     config={
        #         "breaking_strategy": self.config.get("breaking_strategy", "balanced"),
        #         "language": self.config.get("language", "en"),
        #         "prefer_balanced_lines": self.config.get("balance_lines", True),
        #         "respect_sentences": self.config.get("respect_sentences", True),
        #         "min_chars_per_line": self.config.get("min_line_length", 15),
        #         "ideal_chars_per_line": self.config.get("optimal_line_length", 42),
        #         "reading_speed": self.config.get("reading_speed", 20)
        #     }
        # )
    
    def _initialize_emphasis_detection(self):
        """Initialize settings for emphasis detection."""
        # Whether to enable automatic emphasis detection
        self.auto_detect_emphasis = self.config.get("auto_detect_emphasis", True)
        
        # Patterns for detecting emphasis in text
        self.emphasis_patterns = self.config.get("emphasis_patterns", [
            # *word* or _word_ patterns for italic
            (r'\*([\w\s\']+)\*', {'italic': True}),
            (r'_([\w\s\']+)_', {'italic': True}),
            
            # **word** or __word__ patterns for bold
            (r'\*\*([\w\s\']+)\*\*', {'bold': True}),
            (r'__([\w\s\']+)__', {'bold': True}),
            
            # UPPERCASE words (optional, configurable)
            (r'\b([A-Z]{2,})\b', {'bold': True}),
            
            # Custom patterns (can be extended through config)
        ])
        
        # Add any additional patterns from config
        for pattern in self.config.get("additional_emphasis_patterns", []):
            self.emphasis_patterns.append(pattern)
        
        # Keywords that should be emphasized (e.g., "important", "warning")
        self.emphasis_keywords = self.config.get("emphasis_keywords", [
            "important", "critical", "warning", "caution", "note",
            "remember", "key", "essential", "crucial", "vital"
        ])
        
        # Whether to emphasize phrases following emphasis markers
        # e.g., "Note:" or "Important:" would emphasize the text that follows
        self.detect_emphasis_markers = self.config.get("detect_emphasis_markers", True)
        self.emphasis_markers = self.config.get("emphasis_markers", [
            "note:", "important:", "key point:", "remember:", 
            "warning:", "caution:", "attention:"
        ])
    
    def _initialize_reading_speed_settings(self):
        """Initialize reading speed settings from config or defaults."""
        # Default reading speeds in words per minute (WPM) for different profiles
        default_reading_speeds = {
            ReadingSpeedPreset.SLOW: 130,      # 130 WPM - Slow readers
            ReadingSpeedPreset.STANDARD: 170,  # 170 WPM - Average readers
            ReadingSpeedPreset.FAST: 210,      # 210 WPM - Fast readers
            ReadingSpeedPreset.VERY_FAST: 250  # 250 WPM - Very fast readers
        }
        
        # Get reading speeds from config or use defaults
        self.reading_speeds = self.config.get("reading_speeds", default_reading_speeds)
        
        # Default reading speed preset
        self.default_reading_speed_preset = self.config.get(
            "default_reading_speed_preset", 
            ReadingSpeedPreset.STANDARD
        )
        
        # Minimum and maximum subtitle durations (in seconds)
        self.min_subtitle_duration = self.config.get("min_subtitle_duration", 1.0)
        self.max_subtitle_duration = self.config.get("max_subtitle_duration", 7.0)
        
        # Whether to adjust timings when generating subtitles
        self.auto_adjust_timing = self.config.get("auto_adjust_timing", False)
        
        # Character-based adjustment factor (some languages require more time per character)
        self.chars_per_word_equivalent = self.config.get("chars_per_word_equivalent", 5.5)
    
    def _initialize_style_templates(self) -> Dict[str, SubtitleStyle]:
        """
        Initialize built-in style templates.
        
        Returns:
            Dictionary of style templates
        """
        templates = {}
        
        # Default style - clean and readable
        templates["default"] = SubtitleStyle(
            name="Default",
            font_family="Arial",
            font_size=24,
            font_color="#FFFFFF",
            background_color="#00000080",
            bold=False,
            italic=False,
            outline_width=1,
            position=TextPosition.BOTTOM,
            alignment=TextAlignment.CENTER
        )
        
        # Film style - mimics movie subtitles
        templates["film"] = SubtitleStyle(
            name="Film",
            font_family="Helvetica",
            font_size=28,
            font_color="#FFFFFF",
            background_color="#00000000",  # Transparent
            outline_width=2,
            outline_color="#000000",
            shadow_offset=1,
            shadow_color="#000000"
        )
        
        # Documentary style - more academic
        templates["documentary"] = SubtitleStyle(
            name="Documentary",
            font_family="Georgia",
            font_size=24,
            font_color="#FFFFFF",
            background_color="#00000080",
            italic=True,
            max_chars_per_line=36  # Slightly shorter lines for easier reading
        )
        
        # Children's content style - larger, more readable
        templates["children"] = SubtitleStyle(
            name="Children",
            font_family="Comic Sans MS",
            font_size=32,
            font_color="#FFFF00",  # Yellow
            background_color="#00000099",
            bold=True,
            max_lines=1  # One line for easier reading
        )
        
        # TikTok/Instagram style - large, bold, modern
        templates["social_media"] = SubtitleStyle(
            name="SocialMedia",
            font_family="Verdana",
            font_size=36,
            font_color="#FFFFFF",
            background_color="#00000000",
            bold=True,
            outline_width=3,
            outline_color="#000000",
            max_chars_per_line=24  # Short attention spans need shorter lines
        )
        
        # Corporate/Business style - professional
        templates["corporate"] = SubtitleStyle(
            name="Corporate",
            font_family="Calibri",
            font_size=26,
            font_color="#E6E6E6",  # Light gray
            background_color="#00336699",  # Semi-transparent navy
            max_chars_per_line=40
        )
        
        # News style - clear and factual
        templates["news"] = SubtitleStyle(
            name="News",
            font_family="Franklin Gothic",
            font_size=28,
            font_color="#FFFFFF",
            background_color="#00000099",
            bold=True
        )
        
        # Gaming style - vibrant and high-contrast
        templates["gaming"] = SubtitleStyle(
            name="Gaming",
            font_family="Impact",
            font_size=30,
            font_color="#00FF00",  # Bright green
            background_color="#00000099",
            outline_width=2,
            outline_color="#000000"
        )
        
        # Minimal style - unobtrusive
        templates["minimal"] = SubtitleStyle(
            name="Minimal",
            font_family="Arial",
            font_size=20,
            font_color="#CCCCCC",  # Light gray
            background_color="#00000055",  # Very transparent
            max_chars_per_line=46  # Allow longer lines
        )
        
        # Artistic style - stylish and creative
        templates["artistic"] = SubtitleStyle(
            name="Artistic",
            font_family="Garamond",
            font_size=28,
            font_color="#FFD700",  # Gold
            background_color="#00000088",
            italic=True,
            outline_width=1,
            shadow_offset=2
        )
        
        # Educational style - clear, factual
        templates["educational"] = SubtitleStyle(
            name="Educational",
            font_family="Tahoma",
            font_size=26,
            font_color="#FFFFFF",
            background_color="#3366CC99",  # Semi-transparent blue
            bold=True
        )
        
        # Cinematic style - immersive
        templates["cinematic"] = SubtitleStyle(
            name="Cinematic",
            font_family="Palatino Linotype",
            font_size=30,
            font_color="#FFFFFF",
            background_color="#00000000",  # Transparent
            outline_width=2,
            outline_color="#000000",
            shadow_offset=3,
            shadow_color="#000000",
            line_spacing=1.2  # Slightly increased line spacing
        )
        
        # Podcast style - casual conversation
        templates["podcast"] = SubtitleStyle(
            name="Podcast",
            font_family="Segoe UI",
            font_size=26,
            font_color="#FFFFFF",
            background_color="#33333399",  # Dark gray, semi-transparent
            italic=False,
            max_lines=3  # Allow more lines for conversational content
        )
        
        # Accessibility style - maximum readability
        templates["accessibility"] = SubtitleStyle(
            name="Accessibility",
            font_family="Verdana",
            font_size=32,
            font_color="#FFFFFF",
            background_color="#000000CC",  # Nearly opaque black
            bold=True,
            outline_width=2,
            outline_color="#FFFFFF",  # White outline for contrast
            max_chars_per_line=32,  # Shorter lines for readability
            max_lines=2
        )
        
        # Retro style - old school look
        templates["retro"] = SubtitleStyle(
            name="Retro",
            font_family="Courier New",
            font_size=24,
            font_color="#00FF00",  # Green terminal text
            background_color="#000000BB",
            bold=False,
            max_chars_per_line=38
        )
        
        return templates
    
    async def generate_subtitles(
        self,
        transcript: Dict[str, Any],
        output_path: str,
        format: Optional[SubtitleFormat] = None,
        style_name: Optional[str] = None,
        custom_style: Optional[SubtitleStyle] = None,
        reading_speed_preset: Optional[ReadingSpeedPreset] = None,
        adjust_timing: bool = False,
        detect_emphasis: bool = False,
        language: Optional[str] = None,
        auto_detect_language: Optional[bool] = None
    ) -> str:
        """
        Generate subtitle file from transcript.
        
        Args:
            transcript: Transcript with timing information
            output_path: Path to save the subtitle file
            format: Subtitle format to use
            style_name: Name of style template to use
            custom_style: Custom style to use (overrides style_name)
            reading_speed_preset: Reading speed preset for timing adjustment
            adjust_timing: Whether to adjust subtitle timing based on reading speed
            detect_emphasis: Whether to automatically detect and apply formatting to emphasized text
            language: Language code (ISO 639-1) for this subtitle file
            auto_detect_language: Whether to auto-detect language from text
            
        Returns:
            Path to the generated subtitle file
        """
        # Use default format if not specified
        format = format or self.default_format
        
        # Get style to use
        style = self._get_style(style_name, custom_style)
        
        # Determine language to use
        should_detect_language = auto_detect_language if auto_detect_language is not None else self.auto_detect_language
        subtitle_language = language or self.default_language
        
        # Process transcript with language-specific handling
        transcript = self._apply_language_processing(transcript, subtitle_language, should_detect_language, format.value)
        
        # Adjust timing if requested
        if adjust_timing and reading_speed_preset:
            transcript = self.adjust_transcript_timing(transcript, reading_speed_preset)
        
        # Apply emphasis detection if requested
        if detect_emphasis:
            subtitle_format = format.value.lower()
            transcript = self.emphasis_detector.process_transcript(transcript, subtitle_format)
        
        # Generate subtitle content based on format
        if format == SubtitleFormat.SRT:
            subtitle_content = self._generate_srt(transcript, style)
        elif format == SubtitleFormat.VTT:
            subtitle_content = self._generate_vtt(transcript, style)
        elif format == SubtitleFormat.ASS:
            subtitle_content = self._generate_ass(transcript, style)
        elif format == SubtitleFormat.SBV:
            subtitle_content = self._generate_sbv(transcript, style)
        elif format == SubtitleFormat.TTML:
            subtitle_content = self._generate_ttml(transcript, style)
        else:
            raise ValueError(f"Unsupported subtitle format: {format}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Write subtitle content to file
        async with aiofiles.open(output_path, 'w', encoding='utf-8') as file:
            await file.write(subtitle_content)
        
        self.logger.info(f"Generated {format.value} subtitle file at: {output_path}")
        return output_path
    
    def _get_style(
        self, 
        style_name: Optional[str] = None, 
        custom_style: Optional[SubtitleStyle] = None
    ) -> SubtitleStyle:
        """
        Get the subtitle style to use.
        
        Args:
            style_name: Name of style template
            custom_style: Custom style overrides
            
        Returns:
            Subtitle style
        """
        # Start with default style if no specific style requested
        style_name = style_name or self.default_style or "default"
        
        # Get the base style from templates
        if style_name in self.style_templates:
            style = self.style_templates[style_name]
        else:
            self.logger.warning(f"Style template '{style_name}' not found, using default")
            style = self.style_templates["default"]
        
        # Apply custom style overrides if provided
        if custom_style:
            # Create a new style object with base style values
            style_dict = {k: v for k, v in style.__dict__.items()}
            
            # Update with custom style values
            for key, value in custom_style.__dict__.items():
                if value is not None:  # Only override non-None values
                    style_dict[key] = value
            
            # Create a new style with the merged values
            style = SubtitleStyle(**style_dict)
        
        return style
    
    def get_available_styles(self) -> List[str]:
        """
        Get names of all available style templates.
        
        Returns:
            List of style template names
        """
        return list(self.style_templates.keys())
    
    def add_custom_style(self, style: SubtitleStyle) -> None:
        """
        Add a custom style template.
        
        Args:
            style: The style template to add
        """
        self.style_templates[style.name.lower()] = style
        self.logger.info(f"Added custom style template: {style.name}")
    
    async def _generate_srt(
        self,
        transcript: Dict[str, Any],
        style: SubtitleStyle
    ) -> str:
        """
        Generate SRT subtitle file.
        
        Args:
            transcript: Transcript with timing information
            style: Subtitle style (only text formatting is used in SRT)
            
        Returns:
            SRT subtitle file content
        """
        # SRT doesn't support styling, but we use the style for line breaking
        
        segments = transcript.get("segments", [])
        if not segments:
            raise ValueError("Transcript contains no segments")
        
        subtitle_content = ""
        for i, segment in enumerate(segments):
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            
            # Use clean text (without formatting markers) for SRT
            text = segment.get("clean_text", segment.get("text", "")).strip()
            
            if not text:
                continue
            
            # Format timestamps as HH:MM:SS,mmm
            start_formatted = self._format_timestamp_srt(start_time)
            end_formatted = self._format_timestamp_srt(end_time)
            
            # Split text according to style constraints
            text_lines = self._split_text(text, style.max_chars_per_line, style.max_lines)
            
            # Write subtitle entry
            subtitle_content += f"{i+1}\n"
            subtitle_content += f"{start_formatted} --> {end_formatted}\n"
            subtitle_content += "\n".join(text_lines) + "\n\n"
        
        return subtitle_content
    
    async def _generate_vtt(
        self,
        transcript: Dict[str, Any],
        style: SubtitleStyle
    ) -> str:
        """
        Generate WebVTT subtitle file.
        
        Args:
            transcript: Transcript with timing information
            style: Subtitle style
            
        Returns:
            WebVTT subtitle file content
        """
        segments = transcript.get("segments", [])
        if not segments:
            raise ValueError("Transcript contains no segments")
        
        subtitle_content = "WEBVTT\n\n"
        
        # Add style block if supported styles are defined
        css_style = self._style_to_css(style)
        if css_style:
            subtitle_content += "STYLE\n"
            subtitle_content += css_style
            subtitle_content += "\n\n"
        
        for i, segment in enumerate(segments):
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            
            # Get text or formatted text if available
            formatted_parts = segment.get("formatted_parts", None)
            if formatted_parts:
                # VTT supports formatting, so use it
                text = self._format_text_for_vtt(formatted_parts)
            else:
                text = segment.get("text", "").strip()
            
            if not text:
                continue
            
            # Format timestamps as HH:MM:SS.mmm
            start_formatted = self._format_timestamp_vtt(start_time)
            end_formatted = self._format_timestamp_vtt(end_time)
            
            # Split text according to style constraints (if not already formatted)
            if not formatted_parts:
                text_lines = self._split_text(text, style.max_chars_per_line, style.max_lines)
                text = "\n".join(text_lines)
            
            # Write subtitle entry
            subtitle_content += f"cue-{i+1}\n"
            subtitle_content += f"{start_formatted} --> {end_formatted}\n"
            subtitle_content += text + "\n\n"
        
        return subtitle_content
    
    async def _generate_ass(
        self,
        transcript: Dict[str, Any],
        style: SubtitleStyle
    ) -> str:
        """
        Generate ASS subtitle file.
        
        Args:
            transcript: Transcript with timing information
            style: Subtitle style
            
        Returns:
            ASS subtitle file content
        """
        segments = transcript.get("segments", [])
        if not segments:
            raise ValueError("Transcript contains no segments")
        
        subtitle_content = "[Script Info]\n"
        subtitle_content += "Title: Auto-generated subtitles\n"
        subtitle_content += "ScriptType: v4.00+\n"
        subtitle_content += "WrapStyle: 0\n"
        subtitle_content += "ScaledBorderAndShadow: yes\n"
        subtitle_content += "PlayResX: 1920\n"
        subtitle_content += "PlayResY: 1080\n\n"
        
        # Write style section
        subtitle_content += "[V4+ Styles]\n"
        subtitle_content += "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        subtitle_content += style.to_ass_style() + "\n\n"
        
        # Write events section
        subtitle_content += "[Events]\n"
        subtitle_content += "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        
        for segment in segments:
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            text = segment.get("text", "")
            
            # Format times as h:mm:ss.cc
            start_formatted = self._format_ass_time(start_time)
            end_formatted = self._format_ass_time(end_time)
            
            # Format text for ASS (line breaks, etc.)
            text = self._format_text_for_ass(text, style)
            
            # Write subtitle event
            subtitle_content += f"Dialogue: 0,{start_formatted},{end_formatted},{style.name},,0,0,0,,{text}\n"
        
        return subtitle_content
    
    async def _generate_sbv(
        self,
        transcript: Dict[str, Any],
        style: SubtitleStyle
    ) -> str:
        """
        Generate SBV subtitle file.
        
        Args:
            transcript: Transcript with timing information
            style: Subtitle style (most styling is ignored in SBV)
            
        Returns:
            SBV subtitle file content
        """
        segments = transcript.get("segments", [])
        if not segments:
            raise ValueError("Transcript contains no segments")
        
        subtitle_content = ""
        
        for segment in segments:
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            text = segment.get("text", "")
            
            # Format times as H:MM:SS.mmm
            start_formatted = self._format_timestamp_sbv(start_time)
            end_formatted = self._format_timestamp_sbv(end_time)
            
            # Split text according to style constraints
            text_lines = self._split_text(text, style.max_chars_per_line, style.max_lines)
            
            # Write subtitle entry
            subtitle_content += f"{start_formatted},{end_formatted}\n"
            subtitle_content += "\n".join(text_lines) + "\n\n"
        
        return subtitle_content
    
    async def _generate_ttml(
        self,
        transcript: Dict[str, Any],
        style: SubtitleStyle
    ) -> str:
        """
        Generate TTML subtitle file.
        
        Args:
            transcript: Transcript with timing information
            style: Subtitle style
            
        Returns:
            TTML subtitle file content
        """
        segments = transcript.get("segments", [])
        if not segments:
            raise ValueError("Transcript contains no segments")
        
        # Start TTML document
        subtitle_content = '<?xml version="1.0" encoding="UTF-8"?>\n'
        subtitle_content += '<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling">\n'
        subtitle_content += '  <head>\n'
        subtitle_content += '    <styling>\n'
        subtitle_content += f'      <style xml:id="default" tts:fontFamily="{style.font_family}" tts:fontSize="{style.font_size}px" tts:color="{style.font_color}" tts:backgroundColor="{style.background_color}"/>\n'
        
        # Add bold style if needed
        if style.bold:
            subtitle_content += '      <style xml:id="bold" tts:fontWeight="bold"/>\n'
        
        # Add italic style if needed
        if style.italic:
            subtitle_content += '      <style xml:id="italic" tts:fontStyle="italic"/>\n'
        
        subtitle_content += '    </styling>\n'
        subtitle_content += '    <layout>\n'
        subtitle_content += f'      <region xml:id="bottom" tts:origin="10% 80%" tts:extent="80% 20%" tts:textAlign="{style.alignment.value}"/>\n'
        subtitle_content += '    </layout>\n'
        subtitle_content += '  </head>\n'
        subtitle_content += '  <body>\n'
        subtitle_content += '    <div>\n'
        
        for segment in segments:
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            text = segment.get("text", "")
            
            # Format times as HH:MM:SS.MMM
            start_formatted = self._format_timestamp_ttml(start_time)
            end_formatted = self._format_timestamp_ttml(end_time)
            
            # Split text according to style constraints
            text_lines = self._split_text(text, style.max_chars_per_line, style.max_lines)
            
            # Write subtitle entry
            subtitle_content += f'      <p begin="{start_formatted}" end="{end_formatted}" style="default" region="bottom">\n'
            subtitle_content += '        ' + '<br/>'.join(text_lines) + '\n'
            subtitle_content += '      </p>\n'
        
        # Close TTML document
        subtitle_content += '    </div>\n'
        subtitle_content += '  </body>\n'
        subtitle_content += '</tt>'
        
        return subtitle_content
    
    def _split_text(self, text: str, max_chars_per_line: int, max_lines: int) -> List[str]:
        """
        Intelligently split text into lines for optimal readability.
        
        Args:
            text: Text to split
            max_chars_per_line: Maximum characters per line
            max_lines: Maximum number of lines
            
        Returns:
            List of text lines
        """
        # Use the smart text breaker for optimal text splitting
        return self.text_breaker.break_text(
            text=text,
            max_lines=max_lines,
            max_chars_per_line=max_chars_per_line,
            language=self.config.get("language", "en")
        )
    
    def _split_text_advanced(self, text: str, max_chars_per_line: int, max_lines: int) -> List[str]:
        """
        Legacy method kept for backward compatibility.
        Now just calls the smart text breaker.
        """
        return self.text_breaker.break_text(
            text=text,
            max_lines=max_lines,
            max_chars_per_line=max_chars_per_line
        )
    
    def _split_text_basic(self, text: str, max_chars_per_line: int, max_lines: int) -> List[str]:
        """
        Legacy basic text splitting method kept for backward compatibility.
        """
        return self.text_breaker._break_text_basic(
            text=text,
            max_lines=max_lines,
            max_chars_per_line=max_chars_per_line
        )
    
    def _balance_line_lengths(self, lines: List[str], max_chars_per_line: int) -> List[str]:
        """
        Balance line lengths for multi-line subtitles to improve readability.
        Uses the smart text breaker's balancing functionality.
        
        Args:
            lines: Initial line breaks
            max_chars_per_line: Maximum characters per line
            
        Returns:
            Re-balanced lines for more consistent length
        """
        return self.text_breaker._balance_lines(lines, max_chars_per_line)
    
    def _style_to_css(self, style: SubtitleStyle) -> str:
        """
        Convert style to CSS for WebVTT styling.
        
        Args:
            style: Subtitle style
            
        Returns:
            CSS style string
        """
        css = []
        css.append("::cue {")
        css.append(f"  font-family: {style.font_family};")
        css.append(f"  font-size: {style.font_size}px;")
        css.append(f"  color: {style.font_color};")
        css.append(f"  background-color: {style.background_color};")
        
        if style.bold:
            css.append("  font-weight: bold;")
        if style.italic:
            css.append("  font-style: italic;")
        
        # Text alignment
        css.append(f"  text-align: {style.alignment.value};")
        
        css.append("}")
        
        return "\n".join(css)
    
    def _format_timestamp_srt(self, seconds: float) -> str:
        """Format timestamp for SRT format: HH:MM:SS,mmm"""
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{milliseconds:03d}"
    
    def _format_timestamp_vtt(self, seconds: float) -> str:
        """Format timestamp for WebVTT format: HH:MM:SS.mmm"""
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{milliseconds:03d}"
    
    def _format_timestamp_ass(self, seconds: float) -> str:
        """Format timestamp for ASS format: H:MM:SS.cc (centiseconds)"""
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        centiseconds = int((seconds - int(seconds)) * 100)
        return f"{int(hours)}:{int(minutes):02d}:{int(seconds):02d}.{centiseconds:02d}"
    
    def _format_timestamp_sbv(self, seconds: float) -> str:
        """Format timestamp for SBV format: H:MM:SS.mmm"""
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{int(hours)}:{int(minutes):02d}:{int(seconds):02d}.{milliseconds:03d}"
    
    def _format_timestamp_ttml(self, seconds: float) -> str:
        """Format timestamp for TTML format: HH:MM:SS.mmm"""
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{milliseconds:03d}"
    
    def _adjust_subtitle_timing(
        self, 
        transcript: Dict[str, Any], 
        reading_speed_preset: ReadingSpeedPreset
    ) -> Dict[str, Any]:
        """
        Adjust subtitle timing based on reading speed.
        
        Args:
            transcript: Transcript with timing information
            reading_speed_preset: Reading speed preset to use
            
        Returns:
            Transcript with adjusted timing
        """
        # Create a deep copy to avoid modifying the original
        import copy
        adjusted_transcript = copy.deepcopy(transcript)
        
        # Get the words per minute for this preset
        wpm = self.reading_speeds.get(reading_speed_preset, self.reading_speeds[ReadingSpeedPreset.STANDARD])
        
        # Get segments
        segments = adjusted_transcript.get("segments", [])
        if not segments:
            return adjusted_transcript
            
        # For each segment, calculate ideal duration and adjust if necessary
        for i, segment in enumerate(segments):
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            current_duration = end_time - start_time
            
            text = segment.get("text", "").strip()
            if not text:
                continue
                
            # Calculate ideal duration based on text length and reading speed
            ideal_duration = self._calculate_ideal_duration(text, wpm)
            
            # Ensure duration stays within min/max limits
            ideal_duration = max(self.min_subtitle_duration, min(self.max_subtitle_duration, ideal_duration))
            
            # If current duration is significantly shorter than ideal, extend it
            if current_duration < ideal_duration * 0.8:
                # Check if we can extend without overlapping with next segment
                if i < len(segments) - 1:
                    next_start = segments[i+1].get("start", end_time + 10)  # +10 as safety margin
                    max_possible_end = next_start - 0.05  # Small gap between segments
                    new_end = min(start_time + ideal_duration, max_possible_end)
                else:
                    # Last segment, we can extend freely
                    new_end = start_time + ideal_duration
                
                # Update the segment end time
                segment["end"] = new_end
            
            # If current duration is significantly longer than ideal, maybe shorten it
            # Only if auto-shortening is enabled in config
            elif current_duration > ideal_duration * 1.5 and self.config.get("allow_shortening", False):
                # Don't shorten if it's a segment with a natural pause or explicit timing
                if not segment.get("locked_timing", False):
                    segment["end"] = start_time + ideal_duration
        
        return adjusted_transcript
    
    def _calculate_ideal_duration(self, text: str, words_per_minute: int) -> float:
        """
        Calculate ideal subtitle duration based on text content and reading speed.
        
        Args:
            text: Subtitle text
            words_per_minute: Reading speed in words per minute
            
        Returns:
            Ideal duration in seconds
        """
        # Count words (or estimate from characters for non-space languages)
        if ' ' in text:  # Space-separated language like English
            word_count = len(text.split())
        else:  # Character-based language like Chinese
            # Estimate word equivalent based on character count
            word_count = len(text) / self.chars_per_word_equivalent
        
        # Base duration calculation: words / (words per minute / 60)
        # This converts WPM to seconds needed for the given word count
        base_duration = (word_count / words_per_minute) * 60
        
        # Add extra time for processing (constant + percentage of base)
        processing_time = 0.3 + (base_duration * 0.1)  # 0.3s + 10% of base duration
        
        # Calculate final duration with reading ease adjustment
        final_duration = base_duration + processing_time
        
        # Round to 2 decimal places for precision
        return round(final_duration, 2)
    
    def adjust_transcript_timing(
        self,
        transcript: Dict[str, Any],
        reading_speed_preset: ReadingSpeedPreset = ReadingSpeedPreset.STANDARD
    ) -> Dict[str, Any]:
        """
        Public method to adjust subtitle timing based on reading speed.
        
        Args:
            transcript: Transcript with timing information
            reading_speed_preset: Reading speed preset to use
            
        Returns:
            Transcript with adjusted timing
        """
        return self._adjust_subtitle_timing(transcript, reading_speed_preset)
    
    def set_reading_speed(self, preset: ReadingSpeedPreset, words_per_minute: int) -> None:
        """
        Set custom reading speed for a preset.
        
        Args:
            preset: Reading speed preset to modify
            words_per_minute: New reading speed in words per minute
        """
        if words_per_minute < 50 or words_per_minute > 600:
            raise ValueError("Words per minute must be between 50 and 600")
            
        self.reading_speeds[preset] = words_per_minute
        self.logger.info(f"Set reading speed for {preset} to {words_per_minute} WPM")
    
    def get_reading_speed(self, preset: ReadingSpeedPreset) -> int:
        """
        Get reading speed for a preset.
        
        Args:
            preset: Reading speed preset
            
        Returns:
            Reading speed in words per minute
        """
        return self.reading_speeds.get(preset, self.reading_speeds[ReadingSpeedPreset.STANDARD])
    
    def _detect_and_apply_emphasis(self, transcript: Dict[str, Any], style: SubtitleStyle) -> Dict[str, Any]:
        """
        Detect and apply text emphasis using formatting patterns.
        
        Args:
            transcript: Transcript with timing information
            style: Subtitle style to use for formatting decisions
            
        Returns:
            Transcript with emphasis formatting applied
        """
        # Make a deep copy of the transcript to avoid modifying the original
        import copy
        processed_transcript = copy.deepcopy(transcript)
        
        # Process each segment in the transcript
        for segment in processed_transcript.get("segments", []):
            text = segment.get("text", "")
            
            # Skip empty segments
            if not text:
                continue
            
            # Apply emphasis detection and formatting
            formatted_parts = self._process_text_for_emphasis(text, style)
            
            # Store the formatted text in the segment
            segment["formatted_parts"] = formatted_parts
            
            # Also store a version with formatting markers removed for formats that don't support rich text
            segment["clean_text"] = self._get_clean_text(text)
        
        return processed_transcript
    
    def _process_text_for_emphasis(self, text: str, style: SubtitleStyle) -> List[FormattedText]:
        """
        Process text to detect and format emphasized parts.
        
        Args:
            text: Text to process
            style: Subtitle style to use for formatting decisions
            
        Returns:
            List of formatted text parts
        """
        # First, get a clean version of the text with formatting markers removed
        clean_text = self._get_clean_text(text)
        
        # Start with the whole text as a single unformatted part
        parts = [FormattedText(text=clean_text)]
        
        # Apply emphasis patterns
        for pattern, formatting in self.emphasis_patterns:
            new_parts = []
            
            for part in parts:
                # Skip already formatted parts
                if part.bold or part.italic or part.underline or part.color:
                    new_parts.append(part)
                    continue
                
                # Find all matches of the pattern in this part
                matches = list(re.finditer(pattern, part.text))
                
                if not matches:
                    new_parts.append(part)
                    continue
                
                # Split the part at each match
                last_end = 0
                for match in matches:
                    # Add the text before the match
                    if match.start() > last_end:
                        new_parts.append(FormattedText(
                            text=part.text[last_end:match.start()]
                        ))
                    
                    # Add the matched text with formatting
                    emphasis_text = match.group(1) if match.groups() else match.group(0)
                    new_parts.append(FormattedText(
                        text=emphasis_text,
                        bold=formatting.get('bold', False) and style.emphasis_bold,
                        italic=formatting.get('italic', False) and style.emphasis_italic,
                        underline=formatting.get('underline', False),
                        color=formatting.get('color', None) or style.emphasis_color
                    ))
                    
                    last_end = match.end()
                
                # Add any remaining text
                if last_end < len(part.text):
                    new_parts.append(FormattedText(
                        text=part.text[last_end:]
                    ))
            
            # Update parts list with new parts
            parts = new_parts
        
        # Check for emphasis keywords
        if self.emphasis_keywords:
            new_parts = []
            for part in parts:
                # Skip already formatted parts
                if part.bold or part.italic or part.underline or part.color:
                    new_parts.append(part)
                    continue
                
                # Look for emphasis keywords
                words = part.text.split()
                formatted_words = []
                
                for word in words:
                    clean_word = word.lower().strip(".,!?;:()'\"")
                    
                    if clean_word in self.emphasis_keywords:
                        formatted_words.append(FormattedText(
                            text=word,
                            bold=style.emphasis_bold,
                            italic=style.emphasis_italic,
                            color=style.emphasis_color
                        ))
                    else:
                        # If the previous word was not formatted, merge with it
                        if formatted_words and not (formatted_words[-1].bold or formatted_words[-1].italic 
                                                   or formatted_words[-1].underline or formatted_words[-1].color):
                            formatted_words[-1].text += f" {word}"
                        else:
                            formatted_words.append(FormattedText(text=word))
                
                # Merge consecutive unformatted words
                merged_parts = []
                current_part = None
                
                for word_part in formatted_words:
                    if not current_part:
                        current_part = word_part
                    elif (not word_part.bold and not word_part.italic and not word_part.underline and not word_part.color
                         and not current_part.bold and not current_part.italic and not current_part.underline and not current_part.color):
                        current_part.text += f" {word_part.text}"
                    else:
                        merged_parts.append(current_part)
                        current_part = word_part
                
                if current_part:
                    merged_parts.append(current_part)
                
                new_parts.extend(merged_parts)
            
            parts = new_parts
        
        # Check for emphasis markers (e.g., "Note:", "Important:")
        if self.detect_emphasis_markers and self.emphasis_markers:
            new_parts = []
            
            for part in parts:
                # Skip already formatted parts
                if part.bold or part.italic or part.underline or part.color:
                    new_parts.append(part)
                    continue
                
                # Check if the part starts with an emphasis marker
                matched = False
                for marker in self.emphasis_markers:
                    marker_lower = marker.lower()
                    if part.text.lower().startswith(marker_lower):
                        # Split into marker and rest of text
                        marker_text = part.text[:len(marker)]
                        rest_text = part.text[len(marker):].strip()
                        
                        if marker_text:
                            new_parts.append(FormattedText(
                                text=marker_text,
                                bold=style.emphasis_bold,
                                italic=style.emphasis_italic,
                                color=style.emphasis_color
                            ))
                        
                        if rest_text:
                            new_parts.append(FormattedText(
                                text=rest_text,
                                bold=style.emphasis_bold,
                                italic=style.emphasis_italic,
                                color=style.emphasis_color
                            ))
                        
                        matched = True
                        break
                
                if not matched:
                    new_parts.append(part)
            
            parts = new_parts
        
        return parts
    
    def _get_clean_text(self, text: str) -> str:
        """
        Remove formatting markers from text.
        
        Args:
            text: Text with potential formatting markers
            
        Returns:
            Clean text with formatting markers removed
        """
        # Remove markdown-style emphasis markers
        clean = text
        clean = re.sub(r'\*\*([\w\s\']+)\*\*', r'\1', clean)  # **bold**
        clean = re.sub(r'__([\w\s\']+)__', r'\1', clean)      # __bold__
        clean = re.sub(r'\*([\w\s\']+)\*', r'\1', clean)      # *italic*
        clean = re.sub(r'_([\w\s\']+)_', r'\1', clean)        # _italic_
        
        return clean
    
    def _format_text_for_vtt(self, formatted_parts: List[FormattedText]) -> str:
        """
        Format text with WebVTT markup for formatting.
        
        Args:
            formatted_parts: List of formatted text parts
            
        Returns:
            Text with WebVTT formatting markup
        """
        result = []
        
        for part in formatted_parts:
            text = part.text
            
            # Apply formatting
            if part.bold:
                text = f"<b>{text}</b>"
            if part.italic:
                text = f"<i>{text}</i>"
            if part.underline:
                text = f"<u>{text}</u>"
            if part.color:
                text = f'<c.{part.color.lstrip("#")}>{text}</c>'
            
            result.append(text)
        
        return "".join(result)
    
    def _format_text_for_ass(self, text: str, style: SubtitleStyle) -> str:
        """
        Format text for ASS subtitle format.
        
        Args:
            text: Text to format
            style: Subtitle style
            
        Returns:
            Formatted text for ASS subtitle
        """
        # Split text according to style constraints
        text_lines = self._split_text(text, style.max_chars_per_line, style.max_lines)
        
        # Join lines with ASS line break marker
        return "\\N".join(text_lines)
    
    def _format_ass_time(self, seconds: float) -> str:
        """
        Format time for ASS format.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time as h:mm:ss.cc
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        centiseconds = int((seconds - int(seconds)) * 100)
        
        return f"{hours}:{minutes:02d}:{int(seconds):02d}.{centiseconds:02d}" 

    def _apply_language_processing(
        self,
        transcript: Dict[str, Any],
        language_code: str,
        auto_detect: bool,
        format_name: str
    ) -> Dict[str, Any]:
        """
        Apply language-specific processing to transcript.
        
        Args:
            transcript: Original transcript
            language_code: Language code to use
            auto_detect: Whether to auto-detect language
            format_name: Subtitle format name
            
        Returns:
            Processed transcript with language-specific formatting
        """
        # Make a deep copy to avoid modifying the original
        import copy
        processed = copy.deepcopy(transcript)
        
        # Process each segment in the transcript
        for segment in processed.get("segments", []):
            text = segment.get("text", "")
            
            # Skip empty segments
            if not text:
                continue
            
            # Detect language if auto-detection is enabled
            detected_language = None
            if auto_detect:
                detected_language = self.language_support.detect_language(text)
                # If detection failed or returned an unsupported language, fall back
                if not detected_language or not self.language_support.supports_language(detected_language):
                    detected_language = language_code
            else:
                detected_language = language_code
            
            # Apply language-specific formatting
            formatted_text = self.language_support.format_subtitle_for_language(
                text, detected_language, format_name
            )
            
            # Update segment text with language-specific formatting
            segment["text"] = formatted_text
            
            # Store detected language for later use
            segment["language"] = detected_language
            
            # Update font family if not already specified
            if "style" not in segment:
                segment["style"] = {}
            if "font_family" not in segment["style"]:
                font = self.language_support.get_best_font(detected_language)
                segment["style"]["font_family"] = font
        
        return processed

    def _get_style_with_language_support(
        self,
        style: SubtitleStyle,
        language_code: str
    ) -> SubtitleStyle:
        """
        Update style with language-specific font and settings.
        
        Args:
            style: Original style
            language_code: Language code
            
        Returns:
            Updated style with language-specific settings
        """
        # Create a copy of the style to avoid modifying the original
        import copy
        updated_style = copy.deepcopy(style)
        
        # Get recommended font for this language
        recommended_font = self.language_support.get_best_font(language_code)
        
        # Update font family
        updated_style.font_family = recommended_font
        
        # Update text direction if RTL
        direction = self.language_support.get_text_direction(language_code)
        if direction == TextDirection.RTL:
            # For RTL, change alignment if it's not explicitly set
            if not style.custom_style_options.get("explicit_alignment", False):
                updated_style.alignment = TextAlignment.RIGHT
        
        return updated_style

    def _format_vtt_for_language(
        self,
        text: str,
        language_code: str,
        style: SubtitleStyle
    ) -> str:
        """
        Format text for WebVTT with language-specific formatting.
        
        Args:
            text: Original text
            language_code: Language code
            style: Subtitle style
            
        Returns:
            Formatted text with language-specific features
        """
        # Get text direction
        direction = self.language_support.get_text_direction(language_code)
        
        # Normalize text
        text = self.language_support.normalize_text(text, language_code)
        
        # Apply language-specific styles
        if direction == TextDirection.RTL:
            return f'<span dir="rtl">{text}</span>'
        
        return text 