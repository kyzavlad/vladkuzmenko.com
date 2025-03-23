import logging
import re
import math
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class SmartTextBreaker:
    """
    Intelligent text breaking for subtitles with multiple strategies and language support.
    
    This class provides advanced text breaking capabilities for subtitles, including:
    - Basic space-based breaking
    - Linguistic breaking using natural language boundaries
    - Balanced breaking for more even line lengths
    - Support for languages without spaces (CJK languages)
    - Reading complexity analysis
    - Line count optimization
    """
    
    # Punctuation weights for finding optimal break points
    PUNCTUATION_WEIGHTS = {
        '.': 10,  # Period (highest priority)
        '!': 10,  # Exclamation mark
        '?': 10,  # Question mark
        ';': 8,   # Semicolon
        ':': 7,   # Colon
        ',': 5,   # Comma
        '-': 3,   # Dash
        ')': 2,   # Closing parenthesis
        ']': 2,   # Closing bracket
        '}': 2,   # Closing brace
        ' ': 1    # Space (lowest priority)
    }
    
    # CJK Unicode ranges
    CJK_RANGES = [
        (0x4E00, 0x9FFF),    # CJK Unified Ideographs
        (0x3400, 0x4DBF),    # CJK Unified Ideographs Extension A
        (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
        (0x2A700, 0x2B73F),  # CJK Unified Ideographs Extension C
        (0x2B740, 0x2B81F),  # CJK Unified Ideographs Extension D
        (0x2B820, 0x2CEAF),  # CJK Unified Ideographs Extension E
        (0x3300, 0x33FF),    # CJK Compatibility
        (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
        (0xFE30, 0xFE4F),    # CJK Compatibility Forms
        (0xFF00, 0xFFEF),    # Halfwidth and Fullwidth Forms
        (0x2F800, 0x2FA1F)   # CJK Compatibility Ideographs Supplement
    ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the SmartTextBreaker with configuration options.
        
        Args:
            config: Configuration dictionary with options:
                - breaking_strategy: 'basic', 'linguistic', or 'balanced' (default: 'balanced')
                - language: ISO language code (default: 'en')
                - prefer_balanced_lines: Whether to prefer balanced line lengths (default: True)
                - respect_sentences: Whether to avoid breaking sentences (default: True)
                - min_chars_per_line: Minimum characters per line (default: 15)
                - ideal_chars_per_line: Ideal characters per line (default: 42)
                - reading_speed: Average reading speed in chars per second (default: 20)
        """
        self.config = config or {}
        self.breaking_strategy = self.config.get('breaking_strategy', 'balanced')
        self.language = self.config.get('language', 'en')
        self.prefer_balanced_lines = self.config.get('prefer_balanced_lines', True)
        self.respect_sentences = self.config.get('respect_sentences', True)
        self.min_chars_per_line = self.config.get('min_chars_per_line', 15)
        self.ideal_chars_per_line = self.config.get('ideal_chars_per_line', 42)
        self.reading_speed = self.config.get('reading_speed', 20)  # chars per second
        
        # Check if language is CJK
        self.is_cjk_language = self.language in ['zh', 'ja', 'ko']
        
        # Initialize NLTK if needed
        self.nltk_available = False
        if self.breaking_strategy in ['linguistic', 'balanced']:
            self.nltk_available = self._initialize_nltk()
    
    def _initialize_nltk(self) -> bool:
        """
        Initialize NLTK libraries for linguistic text breaking.
        
        Returns:
            bool: True if NLTK is available and initialized, False otherwise
        """
        try:
            import nltk
            # Try to load required NLTK resources
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                try:
                    nltk.download('punkt', quiet=True)
                except Exception as e:
                    logger.warning(f"Failed to download NLTK punkt: {e}")
                    return False
            return True
        except ImportError:
            logger.warning("NLTK not available. Falling back to basic text breaking.")
            return False
    
    def break_text(self, text: str, max_chars_per_line: int = 42, max_lines: int = 2) -> List[str]:
        """
        Break text into multiple lines for subtitles.
        
        Args:
            text: The text to break into lines
            max_chars_per_line: Maximum characters per line
            max_lines: Maximum number of lines
            
        Returns:
            List of text lines
        """
        if not text:
            return []
            
        # If text fits in one line, return it as is
        if len(text) <= max_chars_per_line:
            return [text]
        
        # For CJK languages, use character-based breaking
        if self.is_cjk_language:
            return self._break_cjk_text(text, max_chars_per_line, max_lines)
        
        # Choose breaking strategy
        if self.breaking_strategy == 'linguistic' and self.nltk_available:
            return self._break_text_linguistic(text, max_chars_per_line, max_lines)
        elif self.breaking_strategy == 'balanced':
            return self._break_text_balanced(text, max_chars_per_line, max_lines)
        else:
            return self._break_text_basic(text, max_chars_per_line, max_lines)
    
    def _break_text_basic(self, text: str, max_chars_per_line: int, max_lines: int) -> List[str]:
        """
        Break text using basic space-based algorithm.
        
        Args:
            text: The text to break
            max_chars_per_line: Maximum characters per line
            max_lines: Maximum number of lines
            
        Returns:
            List of text lines
        """
        lines = []
        remaining_text = text
        
        while remaining_text and len(lines) < max_lines:
            if len(remaining_text) <= max_chars_per_line:
                lines.append(remaining_text)
                break
                
            # Find the last space within max_chars_per_line
            break_point = self._find_best_break_point(remaining_text, max_chars_per_line)
            
            if break_point <= 0:
                # No good break point found, force break at max_chars_per_line
                break_point = max_chars_per_line
            
            lines.append(remaining_text[:break_point].strip())
            remaining_text = remaining_text[break_point:].strip()
        
        # If we still have text but reached max_lines, append it to the last line
        if remaining_text and lines:
            lines[-1] = lines[-1] + "..."
        
        return lines
    
    def _break_text_linguistic(self, text: str, max_chars_per_line: int, max_lines: int) -> List[str]:
        """
        Break text using linguistic features (sentences, clauses).
        
        Args:
            text: The text to break
            max_chars_per_line: Maximum characters per line
            max_lines: Maximum number of lines
            
        Returns:
            List of text lines
        """
        try:
            import nltk
            
            # First try to break by sentences
            sentences = nltk.tokenize.sent_tokenize(text)
            
            if len(sentences) >= max_lines:
                # We have enough sentences for our lines
                result = []
                current_line = ""
                
                for sentence in sentences:
                    if len(result) >= max_lines - 1:
                        # Last allowed line, add remaining text
                        if current_line:
                            result.append(current_line)
                        remaining = " ".join([s for s in sentences if s not in " ".join(result)])
                        if len(remaining) > max_chars_per_line:
                            # Need to truncate
                            result.append(remaining[:max_chars_per_line-3] + "...")
                        else:
                            result.append(remaining)
                        break
                        
                    if len(current_line + sentence) <= max_chars_per_line:
                        current_line += sentence + " "
                    else:
                        if current_line:
                            result.append(current_line.strip())
                        current_line = sentence + " "
                
                if current_line and len(result) < max_lines:
                    result.append(current_line.strip())
                    
                return result
            
            # Not enough sentences, try clauses
            clauses = []
            for sentence in sentences:
                # Split by clause markers
                for clause in re.split(r'([,;:])', sentence):
                    if clause in [',', ';', ':']:
                        if clauses:
                            clauses[-1] += clause
                    elif clause.strip():
                        clauses.append(clause.strip())
            
            # If we have enough clauses, use them
            if len(clauses) >= max_lines:
                lines = []
                current_line = ""
                
                for clause in clauses:
                    if len(lines) >= max_lines - 1:
                        # Last allowed line
                        if current_line:
                            lines.append(current_line)
                        remaining = " ".join([c for c in clauses if c not in " ".join(lines)])
                        if len(remaining) > max_chars_per_line:
                            lines.append(remaining[:max_chars_per_line-3] + "...")
                        else:
                            lines.append(remaining)
                        break
                        
                    if len(current_line + clause) <= max_chars_per_line:
                        current_line += clause + " "
                    else:
                        if current_line:
                            lines.append(current_line.strip())
                        current_line = clause + " "
                
                if current_line and len(lines) < max_lines:
                    lines.append(current_line.strip())
                    
                return lines
            
            # Fall back to basic breaking with linguistic break points
            return self._break_text_basic(text, max_chars_per_line, max_lines)
            
        except Exception as e:
            logger.warning(f"Error in linguistic text breaking: {e}")
            return self._break_text_basic(text, max_chars_per_line, max_lines)
    
    def _break_text_balanced(self, text: str, max_chars_per_line: int, max_lines: int) -> List[str]:
        """
        Break text with balanced line lengths.
        
        Args:
            text: The text to break
            max_chars_per_line: Maximum characters per line
            max_lines: Maximum number of lines
            
        Returns:
            List of text lines
        """
        # First get basic breaking
        basic_lines = self._break_text_basic(text, max_chars_per_line, max_lines)
        
        # If we only have one line or reached max_lines, return as is
        if len(basic_lines) <= 1 or len(basic_lines) >= max_lines:
            return basic_lines
        
        # Try to balance line lengths
        total_length = sum(len(line) for line in basic_lines)
        target_length = total_length / len(basic_lines)
        
        # If lines are already reasonably balanced, return as is
        line_lengths = [len(line) for line in basic_lines]
        length_variance = sum((length - target_length) ** 2 for length in line_lengths) / len(line_lengths)
        
        if length_variance < (target_length * 0.2) ** 2:  # Variance threshold
            return basic_lines
        
        # Try to rebalance by finding better break points
        balanced_lines = []
        remaining_text = text
        chars_per_line = int(total_length / max_lines)
        
        for i in range(max_lines - 1):
            if not remaining_text:
                break
                
            # Find a break point near the target length
            break_point = self._find_best_break_point(remaining_text, chars_per_line)
            
            if break_point <= 0:
                # No good break point found, use max_chars_per_line
                break_point = min(len(remaining_text), max_chars_per_line)
            
            balanced_lines.append(remaining_text[:break_point].strip())
            remaining_text = remaining_text[break_point:].strip()
        
        # Add remaining text as the last line
        if remaining_text:
            balanced_lines.append(remaining_text)
        
        # Check if our balancing improved the variance
        balanced_lengths = [len(line) for line in balanced_lines]
        balanced_variance = sum((length - target_length) ** 2 for length in balanced_lengths) / len(balanced_lengths)
        
        return balanced_lines if balanced_variance < length_variance else basic_lines
    
    def _break_cjk_text(self, text: str, max_chars_per_line: int, max_lines: int) -> List[str]:
        """
        Break text for CJK languages (Chinese, Japanese, Korean).
        
        Args:
            text: The text to break
            max_chars_per_line: Maximum characters per line
            max_lines: Maximum number of lines
            
        Returns:
            List of text lines
        """
        lines = []
        remaining_text = text
        
        # CJK punctuation that can be used as break points
        cjk_punctuation = '。，、；：？！""''（）《》【】'
        
        while remaining_text and len(lines) < max_lines:
            if len(remaining_text) <= max_chars_per_line:
                lines.append(remaining_text)
                break
            
            # Try to find a good break point
            best_pos = -1
            
            # First look for CJK punctuation within the limit
            for i in range(max_chars_per_line, 0, -1):
                if i < len(remaining_text) and remaining_text[i] in cjk_punctuation:
                    best_pos = i + 1  # Break after the punctuation
                    break
            
            # If no punctuation found, just break at max_chars_per_line
            if best_pos == -1:
                best_pos = max_chars_per_line
            
            lines.append(remaining_text[:best_pos])
            remaining_text = remaining_text[best_pos:]
        
        # If we still have text but reached max_lines, append it to the last line
        if remaining_text and lines:
            lines[-1] = lines[-1] + "..."
        
        return lines
    
    def _find_best_break_point(self, text: str, max_chars: int) -> int:
        """
        Find the best point to break text based on punctuation and spaces.
        
        Args:
            text: The text to analyze
            max_chars: Maximum characters to consider
            
        Returns:
            Position of the best break point
        """
        if len(text) <= max_chars:
            return len(text)
        
        # Limit our search to max_chars
        search_text = text[:max_chars+1]
        
        # Find all potential break points with their weights
        break_points = []
        
        for i in range(len(search_text) - 1, 0, -1):
            char = search_text[i]
            prev_char = search_text[i-1] if i > 0 else ""
            
            # Check for punctuation followed by space
            if prev_char in self.PUNCTUATION_WEIGHTS and char == ' ':
                weight = self.PUNCTUATION_WEIGHTS[prev_char]
                break_points.append((i+1, weight))  # Position after the space
            # Check for space
            elif char == ' ':
                break_points.append((i+1, self.PUNCTUATION_WEIGHTS[' ']))
        
        # Sort by weight (highest first) and position (furthest first)
        break_points.sort(key=lambda x: (-x[1], -x[0]))
        
        # Return the best break point, or 0 if none found
        return break_points[0][0] if break_points else 0
    
    def analyze_reading_complexity(self, text: str) -> Dict[str, Any]:
        """
        Analyze the reading complexity of text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with complexity metrics
        """
        if not text:
            return {"complexity": "unknown", "avg_word_length": 0, "sentence_count": 0}
        
        result = {
            "complexity": "unknown",
            "avg_word_length": 0,
            "sentence_count": 0,
            "word_count": 0,
            "char_count": len(text),
            "estimated_read_time_ms": int(len(text) / self.reading_speed * 1000)
        }
        
        # Basic analysis without NLTK
        words = text.split()
        result["word_count"] = len(words)
        
        if words:
            result["avg_word_length"] = sum(len(word) for word in words) / len(words)
        
        # More detailed analysis with NLTK if available
        if self.nltk_available:
            try:
                import nltk
                
                # Count sentences
                sentences = nltk.tokenize.sent_tokenize(text)
                result["sentence_count"] = len(sentences)
                
                # Get more accurate word count
                words = nltk.tokenize.word_tokenize(text)
                result["word_count"] = len(words)
                
                if words:
                    result["avg_word_length"] = sum(len(word) for word in words if word.isalnum()) / len(words)
                
                # Calculate complexity based on word length and sentence length
                if result["sentence_count"] > 0:
                    words_per_sentence = result["word_count"] / result["sentence_count"]
                    complexity_score = (result["avg_word_length"] * 0.6) + (words_per_sentence * 0.4)
                    
                    if complexity_score < 4:
                        result["complexity"] = "simple"
                    elif complexity_score < 5:
                        result["complexity"] = "moderate"
                    else:
                        result["complexity"] = "complex"
                else:
                    result["complexity"] = "simple"
                    
            except Exception as e:
                logger.warning(f"Error in complexity analysis: {e}")
        
        return result
    
    def optimize_line_count(self, text: str, max_lines: int = 2) -> int:
        """
        Determine the optimal number of lines for a given text.
        
        Args:
            text: The text to analyze
            max_lines: Maximum number of lines allowed
            
        Returns:
            Optimal number of lines (1 to max_lines)
        """
        if not text:
            return 1
            
        # For very short text, use one line
        if len(text) < self.min_chars_per_line:
            return 1
            
        # For very long text, use max lines
        if len(text) > self.ideal_chars_per_line * max_lines:
            return max_lines
            
        # Analyze complexity
        complexity = self.analyze_reading_complexity(text)
        
        # Calculate optimal line count based on text length and complexity
        optimal_chars_per_line = self.ideal_chars_per_line
        
        if complexity["complexity"] == "complex":
            # Use shorter lines for complex text
            optimal_chars_per_line = int(self.ideal_chars_per_line * 0.8)
        elif complexity["complexity"] == "simple":
            # Can use longer lines for simple text
            optimal_chars_per_line = int(self.ideal_chars_per_line * 1.2)
            
        # Calculate lines needed
        lines_needed = math.ceil(len(text) / optimal_chars_per_line)
        
        # Constrain to max_lines
        return min(lines_needed, max_lines)
    
    def is_cjk_char(self, char: str) -> bool:
        """
        Check if a character is a CJK (Chinese, Japanese, Korean) character.
        
        Args:
            char: The character to check
            
        Returns:
            True if the character is CJK, False otherwise
        """
        if not char:
            return False
            
        code_point = ord(char)
        
        for start, end in self.CJK_RANGES:
            if start <= code_point <= end:
                return True
                
        return False
        
    def _balance_lines(self, lines: List[str], max_chars_per_line: int) -> List[str]:
        """
        Balance line lengths for multi-line subtitles to improve readability.
        
        Args:
            lines: Initial line breaks
            max_chars_per_line: Maximum characters per line
            
        Returns:
            Re-balanced lines for more consistent length
        """
        if not lines or len(lines) <= 1:
            return lines
            
        # Calculate current line length statistics
        line_lengths = [len(line) for line in lines]
        total_length = sum(line_lengths)
        target_length = total_length / len(lines)
        
        # Calculate variance to measure imbalance
        length_variance = sum((length - target_length) ** 2 for length in line_lengths) / len(line_lengths)
        
        # If lines are already reasonably balanced, return as is
        if length_variance < (target_length * 0.2) ** 2:  # Variance threshold
            return lines
            
        # Try to rebalance by using the balanced breaking strategy
        combined_text = " ".join(lines)
        return self._break_text_balanced(combined_text, max_chars_per_line, len(lines)) 