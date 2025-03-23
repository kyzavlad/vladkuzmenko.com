import logging
import re
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple, Set

logger = logging.getLogger(__name__)

class EmphasisFormat(str, Enum):
    """Types of emphasis formatting available for subtitles."""
    BOLD = "bold"
    ITALIC = "italic"
    UNDERLINE = "underline"
    COLOR = "color"
    NONE = "none"

class EmphasisPattern(str, Enum):
    """Common patterns used to identify emphasis in text."""
    MARKDOWN_BOLD = r'\*\*(.*?)\*\*'           # **word**
    MARKDOWN_ITALIC = r'\*(.*?)\*'             # *word*
    MARKDOWN_UNDERSCORE_BOLD = r'__(.*?)__'    # __word__
    MARKDOWN_UNDERSCORE_ITALIC = r'_(.*?)_'    # _word_
    UPPERCASE_WORD = r'\b[A-Z][A-Z]+\b'        # WORD
    QUOTES = r'"([^"]*)"'                      # "word"
    

class EmphasisDetector:
    """
    Detects and applies emphasis formatting to subtitle text.
    
    Features:
    - Multiple detection methods (markdown-like, uppercase, key phrases)
    - Format-specific rendering (VTT, ASS, TTML)
    - Customizable emphasis styles
    - Support for nested formatting
    """
    
    # Format-specific tags for various subtitle formats
    FORMAT_TAGS = {
        'vtt': {
            EmphasisFormat.BOLD: ('<b>', '</b>'),
            EmphasisFormat.ITALIC: ('<i>', '</i>'),
            EmphasisFormat.UNDERLINE: ('<u>', '</u>'),
            EmphasisFormat.COLOR: ('<c.{color}>', '</c>')
        },
        'ass': {
            EmphasisFormat.BOLD: ('{\\b1}', '{\\b0}'),
            EmphasisFormat.ITALIC: ('{\\i1}', '{\\i0}'),
            EmphasisFormat.UNDERLINE: ('{\\u1}', '{\\u0}'),
            EmphasisFormat.COLOR: ('{\\c&H{color}&}', '{\\c}')
        },
        'ttml': {
            EmphasisFormat.BOLD: ('<span fontWeight="bold">', '</span>'),
            EmphasisFormat.ITALIC: ('<span fontStyle="italic">', '</span>'),
            EmphasisFormat.UNDERLINE: ('<span textDecoration="underline">', '</span>'),
            EmphasisFormat.COLOR: ('<span color="{color}">', '</span>')
        },
        'srt': {
            EmphasisFormat.BOLD: ('<b>', '</b>'),
            EmphasisFormat.ITALIC: ('<i>', '</i>'),
            EmphasisFormat.UNDERLINE: ('<u>', '</u>'),
            EmphasisFormat.COLOR: ('<font color="{color}">', '</font>')
        }
    }
    
    # Key phrases that often indicate emphasis
    DEFAULT_KEY_PHRASES = [
        "important",
        "warning",
        "caution",
        "remember",
        "note",
        "crucial",
        "essential",
        "critical",
        "urgent",
        "never",
        "always",
        "must",
        "key",
        "vital"
    ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the emphasis detector with configuration options.
        
        Args:
            config: Configuration dictionary with options:
                - detect_markdown: Detect markdown-style formatting (default: True)
                - detect_uppercase: Detect uppercase words for emphasis (default: True)
                - detect_key_phrases: Detect key phrases for emphasis (default: True)
                - detect_quotes: Detect quoted text for emphasis (default: False)
                - custom_key_phrases: List of additional key phrases to emphasize
                - clean_markdown: Remove markdown markers after detection (default: True)
                - emphasis_format: Default emphasis format to apply (default: 'bold')
                - key_phrase_format: Format for key phrases (default: 'italic')
                - uppercase_format: Format for uppercase words (default: 'bold')
                - quote_format: Format for quoted text (default: 'italic')
                - emphasis_color: Color for emphasized text (default: None)
        """
        self.config = config or {}
        
        # Set detection flags
        self.detect_markdown = self.config.get('detect_markdown', True)
        self.detect_uppercase = self.config.get('detect_uppercase', True)
        self.detect_key_phrases = self.config.get('detect_key_phrases', True)
        self.detect_quotes = self.config.get('detect_quotes', False)
        self.clean_markdown = self.config.get('clean_markdown', True)
        
        # Set emphasis formats
        self.emphasis_format = self.config.get('emphasis_format', EmphasisFormat.BOLD)
        self.key_phrase_format = self.config.get('key_phrase_format', EmphasisFormat.ITALIC)
        self.uppercase_format = self.config.get('uppercase_format', EmphasisFormat.BOLD)
        self.quote_format = self.config.get('quote_format', EmphasisFormat.ITALIC)
        
        # Set emphasis color (if used)
        self.emphasis_color = self.config.get('emphasis_color', None)
        
        # Set key phrases
        custom_phrases = self.config.get('custom_key_phrases', [])
        self.key_phrases = set(self.DEFAULT_KEY_PHRASES + custom_phrases)
        
        # Other patterns
        self.emphasis_patterns = {
            'markdown_bold': EmphasisPattern.MARKDOWN_BOLD,
            'markdown_italic': EmphasisPattern.MARKDOWN_ITALIC,
            'markdown_underscore_bold': EmphasisPattern.MARKDOWN_UNDERSCORE_BOLD,
            'markdown_underscore_italic': EmphasisPattern.MARKDOWN_UNDERSCORE_ITALIC,
            'uppercase': EmphasisPattern.UPPERCASE_WORD,
            'quotes': EmphasisPattern.QUOTES
        }
    
    def detect_emphasis(self, text: str) -> List[Tuple[int, int, EmphasisFormat]]:
        """
        Detect segments of text that should be emphasized.
        
        Args:
            text: Subtitle text to analyze
            
        Returns:
            List of tuples with (start_index, end_index, emphasis_format)
        """
        emphasis_regions = []
        
        # Detect markdown-style formatting
        if self.detect_markdown:
            # Bold with ** markers
            for match in re.finditer(self.emphasis_patterns['markdown_bold'], text):
                emphasis_regions.append((match.start(1), match.end(1), EmphasisFormat.BOLD))
            
            # Italic with * markers
            for match in re.finditer(self.emphasis_patterns['markdown_italic'], text):
                # Skip if already part of a bold match
                if not any(start <= match.start(1) <= end for start, end, _ in emphasis_regions):
                    emphasis_regions.append((match.start(1), match.end(1), EmphasisFormat.ITALIC))
            
            # Bold with __ markers
            for match in re.finditer(self.emphasis_patterns['markdown_underscore_bold'], text):
                emphasis_regions.append((match.start(1), match.end(1), EmphasisFormat.BOLD))
            
            # Italic with _ markers
            for match in re.finditer(self.emphasis_patterns['markdown_underscore_italic'], text):
                # Skip if already part of a bold match
                if not any(start <= match.start(1) <= end for start, end, _ in emphasis_regions):
                    emphasis_regions.append((match.start(1), match.end(1), EmphasisFormat.ITALIC))
        
        # Detect uppercase words
        if self.detect_uppercase:
            for match in re.finditer(self.emphasis_patterns['uppercase'], text):
                # Only apply if the word is at least 2 characters
                if match.end() - match.start() >= 2:
                    # Skip common acronyms and abbreviations
                    word = text[match.start():match.end()]
                    if not self._is_common_acronym(word):
                        emphasis_regions.append((match.start(), match.end(), self.uppercase_format))
        
        # Detect key phrases
        if self.detect_key_phrases:
            words = text.lower().split()
            for i, word in enumerate(words):
                # Clean the word of punctuation for matching
                clean_word = re.sub(r'[^\w]', '', word)
                if clean_word in self.key_phrases:
                    # Find position of this word in the original text
                    # This is a simplified approach and might not be accurate for all cases
                    start_pos = 0
                    for j in range(i):
                        start_pos = text.lower().find(words[j], start_pos) + len(words[j])
                        start_pos = text.lower().find(' ', start_pos) + 1
                    end_pos = text.lower().find(word, start_pos) + len(word)
                    emphasis_regions.append((start_pos, end_pos, self.key_phrase_format))
        
        # Detect quoted text
        if self.detect_quotes:
            for match in re.finditer(self.emphasis_patterns['quotes'], text):
                emphasis_regions.append((match.start(1), match.end(1), self.quote_format))
        
        return emphasis_regions
    
    def _is_common_acronym(self, word: str) -> bool:
        """Check if a word is a common acronym or abbreviation."""
        # List of common acronyms and abbreviations
        common_acronyms = {
            "TV", "CEO", "CFO", "CTO", "DVD", "FBI", "CIA", "NASA", "USA", "UK", "EU",
            "UN", "WHO", "GPS", "HTTP", "HTML", "CSS", "API", "URL", "ID", "USB", "HD",
            "AM", "PM", "GMT", "PhD", "MD", "BA", "BS", "MS"
        }
        return word in common_acronyms
    
    def apply_emphasis(self, text: str, subtitle_format: str = 'vtt') -> str:
        """
        Apply emphasis formatting to the text.
        
        Args:
            text: Text to format
            subtitle_format: Subtitle format to use for formatting tags
            
        Returns:
            Formatted text with emphasis tags
        """
        # Normalize subtitle format
        subtitle_format = subtitle_format.lower()
        if subtitle_format not in self.FORMAT_TAGS:
            logger.warning(f"Unsupported subtitle format: {subtitle_format}, using VTT")
            subtitle_format = 'vtt'
        
        # Get the format tags for this subtitle format
        format_tags = self.FORMAT_TAGS[subtitle_format]
        
        # Detect emphasis regions
        emphasis_regions = self.detect_emphasis(text)
        
        # Sort regions by start position (in reverse order to preserve indices)
        emphasis_regions.sort(key=lambda x: x[0], reverse=True)
        
        # Create a copy of the text for formatting
        formatted_text = text
        
        # Apply formatting tags to each region
        for start, end, format_type in emphasis_regions:
            if format_type == EmphasisFormat.COLOR and self.emphasis_color:
                opening_tag = format_tags[format_type][0].format(color=self.emphasis_color)
            else:
                opening_tag = format_tags[format_type][0]
            
            closing_tag = format_tags[format_type][1]
            
            # Insert tags
            formatted_text = (
                formatted_text[:end] + closing_tag +
                formatted_text[end:start] + opening_tag +
                formatted_text[start:]
            )
        
        # Clean markdown markers if needed
        if self.detect_markdown and self.clean_markdown:
            # Remove markdown markers
            formatted_text = re.sub(r'\*\*(.*?)\*\*', r'\1', formatted_text)
            formatted_text = re.sub(r'\*(.*?)\*', r'\1', formatted_text)
            formatted_text = re.sub(r'__(.*?)__', r'\1', formatted_text)
            formatted_text = re.sub(r'_(.*?)_', r'\1', formatted_text)
        
        return formatted_text
    
    def process_transcript(
        self, 
        transcript: Dict[str, Any], 
        subtitle_format: str = 'vtt'
    ) -> Dict[str, Any]:
        """
        Process a transcript to add emphasis formatting.
        
        Args:
            transcript: Transcript with timing information
            subtitle_format: Subtitle format to use for formatting tags
            
        Returns:
            Updated transcript with emphasis formatting
        """
        # Create a deep copy of the transcript
        updated_transcript = transcript.copy()
        
        # Process each segment
        if 'segments' in updated_transcript:
            for segment in updated_transcript['segments']:
                if 'text' in segment:
                    segment['text'] = self.apply_emphasis(segment['text'], subtitle_format)
        
        return updated_transcript
    
    def set_emphasis_format(self, format_type: EmphasisFormat) -> None:
        """
        Set the default emphasis format.
        
        Args:
            format_type: Emphasis format to use
        """
        self.emphasis_format = format_type
    
    def set_emphasis_color(self, color: str) -> None:
        """
        Set the emphasis color.
        
        Args:
            color: Color for emphasized text (hex code or name)
        """
        self.emphasis_color = color
    
    def add_key_phrases(self, phrases: List[str]) -> None:
        """
        Add custom key phrases to emphasize.
        
        Args:
            phrases: List of phrases to add
        """
        self.key_phrases.update(phrases)
    
    def enable_all_detection(self) -> None:
        """Enable all emphasis detection methods."""
        self.detect_markdown = True
        self.detect_uppercase = True
        self.detect_key_phrases = True
        self.detect_quotes = True
    
    def disable_all_detection(self) -> None:
        """Disable all emphasis detection methods."""
        self.detect_markdown = False
        self.detect_uppercase = False
        self.detect_key_phrases = False
        self.detect_quotes = False 