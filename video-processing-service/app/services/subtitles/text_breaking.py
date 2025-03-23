import re
import logging
from typing import List, Dict, Any, Optional, Tuple
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

class SmartTextBreaker:
    """
    Smart text breaking service for subtitles that respects linguistic patterns.
    
    Features:
    - Sentence-aware line breaking
    - Natural language parsing for optimal break points
    - Balanced line lengths for better readability
    - Language-specific processing
    - Configurable breaking strategies
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the text breaker with options.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLTK resources if available
        self.nltk_available = self._initialize_nltk()
        
        # Set breaking strategy
        self.breaking_strategy = self.config.get('breaking_strategy', 'linguistic')  # linguistic, basic, balanced
        
        # Configure line length preferences
        self.optimal_line_length = self.config.get('optimal_line_length', 42)
        self.max_line_length = self.config.get('max_line_length', 50)
        self.min_line_length = self.config.get('min_line_length', 15)
        
        # Line balancing preferences
        self.balance_lines = self.config.get('balance_lines', True)
        self.balance_threshold = self.config.get('balance_threshold', 0.5)  # Percentage difference to trigger balancing
        
        # Configure break point preferences (higher priority first)
        self.break_points = self.config.get('break_points', [
            r'(?<=[.!?])\s+',           # End of sentences
            r'(?<=[:;])\s+',            # After colons and semicolons
            r'(?<=,)\s+',               # After commas
            r'(?<=[a-zA-Z0-9])\s+(?=and|or|but|because|however|therefore)\s+',  # Before conjunctions
            r'(?<=[a-zA-Z0-9])\s+-\s+', # Around em dashes
            r'(?<=\))\s+',              # After closing parenthesis
            r'\s+(?=\()',               # Before opening parenthesis
            r'\s+'                      # Any space as a last resort
        ])
        
        # Language-specific config
        self.language = self.config.get('language', 'en')
        self.languages_without_spaces = ['zh', 'ja', 'ko', 'th']
        self.chars_per_word_equivalent = self.config.get('chars_per_word_equivalent', 5.5)
        
        # Hyphenation settings
        self.enable_hyphenation = self.config.get('enable_hyphenation', False)
        
        # Whether to join short lines if possible
        self.join_short_lines = self.config.get('join_short_lines', True)
    
    def _initialize_nltk(self) -> bool:
        """
        Initialize NLTK resources for text processing.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if punkt is available
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            return True
        except Exception as e:
            self.logger.warning(f"NLTK initialization failed, using basic text splitting: {str(e)}")
            return False
    
    def break_text(
        self,
        text: str,
        max_lines: int = 2,
        max_chars_per_line: Optional[int] = None,
        language: Optional[str] = None
    ) -> List[str]:
        """
        Break text into optimal subtitle lines.
        
        Args:
            text: Text to break into lines
            max_lines: Maximum number of subtitle lines
            max_chars_per_line: Maximum characters per line (overrides default)
            language: Language code (overrides default)
            
        Returns:
            List of subtitle lines
        """
        if not text:
            return []
            
        # Override defaults if provided
        max_chars_per_line = max_chars_per_line or self.max_line_length
        language = language or self.language
        
        # If text is already short enough, return as a single line
        if len(text) <= max_chars_per_line:
            return [text]
        
        # Choose breaking strategy based on language and config
        if language in self.languages_without_spaces:
            return self._break_text_without_spaces(text, max_lines, max_chars_per_line, language)
        elif self.breaking_strategy == 'linguistic' and self.nltk_available:
            return self._break_text_linguistic(text, max_lines, max_chars_per_line)
        elif self.breaking_strategy == 'balanced':
            return self._break_text_balanced(text, max_lines, max_chars_per_line)
        else:
            return self._break_text_basic(text, max_lines, max_chars_per_line)
    
    def _break_text_linguistic(self, text: str, max_lines: int, max_chars_per_line: int) -> List[str]:
        """
        Break text using linguistic analysis.
        
        Args:
            text: Text to break into lines
            max_lines: Maximum number of subtitle lines
            max_chars_per_line: Maximum characters per line
            
        Returns:
            List of subtitle lines
        """
        try:
            # First check if we can use sentence boundaries
            sentences = sent_tokenize(text)
            
            # If we have just the right number of sentences and they fit
            if 1 <= len(sentences) <= max_lines and all(len(sent) <= max_chars_per_line for sent in sentences):
                return sentences
            
            # If sentences are too long or too many, we need to break them further
            lines = []
            current_sentence_index = 0
            
            while len(lines) < max_lines and current_sentence_index < len(sentences):
                current_sentence = sentences[current_sentence_index]
                
                # If the sentence fits on one line, add it and move to the next
                if len(current_sentence) <= max_chars_per_line:
                    lines.append(current_sentence)
                    current_sentence_index += 1
                    continue
                
                # Otherwise, need to break the sentence
                best_break_point = self._find_best_break_point(current_sentence, max_chars_per_line)
                
                if best_break_point > 0:
                    lines.append(current_sentence[:best_break_point].strip())
                    
                    # Update the current sentence to be the remainder
                    sentences[current_sentence_index] = current_sentence[best_break_point:].strip()
                else:
                    # No good break point found, force break at max_chars_per_line
                    lines.append(current_sentence[:max_chars_per_line].strip())
                    sentences[current_sentence_index] = current_sentence[max_chars_per_line:].strip()
                
                # If we've used all the lines, break
                if len(lines) >= max_lines:
                    break
                
                # If the remainder of the sentence is empty, move to the next sentence
                if not sentences[current_sentence_index]:
                    current_sentence_index += 1
            
            # Handle case where we didn't use all lines but have more sentences
            # Try to fit more sentences on the last line if they're short
            if len(lines) < max_lines and current_sentence_index < len(sentences):
                next_sentence = sentences[current_sentence_index]
                
                # Only join if the combined length is reasonable
                if lines and len(lines[-1]) + len(next_sentence) + 1 <= max_chars_per_line:
                    lines[-1] = f"{lines[-1]} {next_sentence}"
                    current_sentence_index += 1
            
            # If we have too much text, add ellipsis to indicate more content
            if current_sentence_index < len(sentences) or (current_sentence_index == len(sentences) - 1 and sentences[current_sentence_index]):
                if lines[-1].endswith('...'):
                    pass  # Already has ellipsis
                elif lines[-1].endswith('.'):
                    lines[-1] = f"{lines[-1].rstrip('.')}..."
                else:
                    lines[-1] = f"{lines[-1]}..."
            
            # Balance line lengths if needed
            if self.balance_lines and len(lines) > 1:
                lines = self._balance_lines(lines, max_chars_per_line)
            
            return lines
            
        except Exception as e:
            self.logger.warning(f"Linguistic text breaking failed: {str(e)}")
            return self._break_text_basic(text, max_lines, max_chars_per_line)
    
    def _find_best_break_point(self, text: str, max_chars: int) -> int:
        """
        Find the best point to break text based on linguistic patterns.
        
        Args:
            text: Text to analyze for break points
            max_chars: Maximum characters for the first line
            
        Returns:
            Index of the best break point, or -1 if no good break found
        """
        if len(text) <= max_chars:
            return len(text)
        
        # Limit text search to max_chars + a buffer for better breaks
        search_text = text[:min(len(text), max_chars + 15)]
        
        # Try each break pattern in order of preference
        for pattern in self.break_points:
            # Find all matches of this pattern
            matches = list(re.finditer(pattern, search_text))
            
            # Find the last match that's within our line length limit
            for match in reversed(matches):
                if match.end() <= max_chars:
                    return match.end()
        
        # If no good break found, look for any space within the limit
        last_space = search_text[:max_chars].rfind(' ')
        if last_space > self.min_line_length:  # Only use if not too short
            return last_space + 1
        
        # Fallback to character counting if all else fails
        # We'll try not to break words if possible
        if max_chars < len(text) and text[max_chars] != ' ' and ' ' in text[:max_chars]:
            return text[:max_chars].rstrip().rfind(' ') + 1
        
        # Absolute fallback: just break at max_chars
        return max_chars
    
    def _break_text_basic(self, text: str, max_lines: int, max_chars_per_line: int) -> List[str]:
        """
        Basic space-based text breaking.
        
        Args:
            text: Text to break into lines
            max_lines: Maximum number of subtitle lines
            max_chars_per_line: Maximum characters per line
            
        Returns:
            List of subtitle lines
        """
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            # Length of the current line + space + new word
            potential_length = len(current_line) + (1 if current_line else 0) + len(word)
            
            if potential_length <= max_chars_per_line:
                # Add word to current line
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
            else:
                # Line would be too long, start a new line
                lines.append(current_line)
                current_line = word
                
                # Check if we've hit the maximum number of lines
                if len(lines) >= max_lines - 1:  # -1 to reserve space for current_line
                    break
        
        # Add the last line if there's content
        if current_line:
            lines.append(current_line)
        
        # If we've truncated the text, add ellipsis to the last line
        if len(lines) == max_lines and word != words[-1]:
            lines[-1] = lines[-1].rstrip() + "..."
        
        return lines
    
    def _break_text_balanced(self, text: str, max_lines: int, max_chars_per_line: int) -> List[str]:
        """
        Break text with balanced line lengths.
        
        Args:
            text: Text to break into lines
            max_lines: Maximum number of subtitle lines
            max_chars_per_line: Maximum characters per line
            
        Returns:
            List of subtitle lines with balanced lengths
        """
        # Start with basic or linguistic breaking (whatever is available)
        if self.nltk_available:
            lines = self._break_text_linguistic(text, max_lines, max_chars_per_line)
        else:
            lines = self._break_text_basic(text, max_lines, max_chars_per_line)
        
        # Now balance the lines
        return self._balance_lines(lines, max_chars_per_line)
    
    def _balance_lines(self, lines: List[str], max_chars_per_line: int) -> List[str]:
        """
        Balance the length of lines for more even subtitle display.
        
        Args:
            lines: Existing line breaks
            max_chars_per_line: Maximum characters per line
            
        Returns:
            Re-balanced lines
        """
        if len(lines) <= 1:
            return lines
        
        # Check if lines need balancing
        line_lengths = [len(line) for line in lines]
        max_length = max(line_lengths)
        min_length = min(line_lengths)
        
        # If line lengths are already similar, don't adjust
        if max_length - min_length <= max_length * self.balance_threshold:
            return lines
        
        # Case 1: Two lines with very unbalanced lengths
        if len(lines) == 2:
            # Try to find a break point that balances the lines
            total_text = " ".join(lines)
            target_length = len(total_text) // 2
            
            # Look for a break point near the middle
            search_window = int(len(total_text) * 0.2)  # 20% search window around the middle
            search_start = max(0, target_length - search_window)
            search_end = min(len(total_text), target_length + search_window)
            
            # Find the best break point in this region
            search_text = total_text[search_start:search_end]
            
            # Try each break pattern
            best_break = -1
            for pattern in self.break_points:
                matches = list(re.finditer(pattern, search_text))
                if matches:
                    # Find the break closest to the target length
                    closest_match = min(matches, key=lambda m: abs((m.end() + search_start) - target_length))
                    best_break = closest_match.end() + search_start
                    break
            
            # If we found a good break, use it
            if best_break > 0:
                return [total_text[:best_break].strip(), total_text[best_break:].strip()]
            
            # Fallback: just split at a space near the middle
            spaces = [i for i, char in enumerate(total_text) if char == ' ']
            if spaces:
                best_space = min(spaces, key=lambda i: abs(i - target_length))
                return [total_text[:best_space].strip(), total_text[best_space+1:].strip()]
        
        # For more than 2 lines, try to balance by adjusting break points
        if len(lines) > 2:
            # Join all text and re-break with a more even distribution
            total_text = " ".join(lines)
            target_length = len(total_text) // len(lines)
            
            balanced_lines = []
            remaining_text = total_text
            
            while len(balanced_lines) < len(lines) - 1 and remaining_text:
                # Find a break point near the target length
                break_point = self._find_best_break_point(remaining_text, target_length)
                
                if break_point > 0:
                    balanced_lines.append(remaining_text[:break_point].strip())
                    remaining_text = remaining_text[break_point:].strip()
                else:
                    # Fall back to basic splitting if we can't find a good break
                    break
            
            # Add the remaining text as the last line
            if remaining_text:
                balanced_lines.append(remaining_text)
            
            # Make sure we have the right number of lines
            while len(balanced_lines) < len(lines):
                balanced_lines.append("")
            
            # Check if any line exceeds max length
            if any(len(line) > max_chars_per_line for line in balanced_lines):
                return lines  # Revert to original if new lines are too long
            
            return balanced_lines[:len(lines)]
        
        return lines
    
    def _break_text_without_spaces(
        self,
        text: str,
        max_lines: int,
        max_chars_per_line: int,
        language: str
    ) -> List[str]:
        """
        Special text breaking for languages without spaces (CJK).
        
        Args:
            text: Text to break into lines
            max_lines: Maximum number of subtitle lines
            max_chars_per_line: Maximum characters per line
            language: Language code
            
        Returns:
            List of subtitle lines
        """
        # For CJK languages, we can simply chunk by character count
        lines = []
        
        # Look for natural breaks in CJK text (periods, commas, etc.)
        # Common CJK punctuation marks
        cjk_punctuation = "。，、；：？！""''（）《》「」『』【】〔〕…─"
        
        # First check if any natural sentence breaks fall within our limits
        natural_breaks = []
        for i, char in enumerate(text):
            if char in cjk_punctuation and i < len(text) - 1:
                natural_breaks.append(i + 1)
        
        # Try to use these natural breaks for line splitting
        if natural_breaks:
            current_pos = 0
            for break_pos in natural_breaks:
                segment_length = break_pos - current_pos
                
                if segment_length <= max_chars_per_line:
                    # This segment fits on one line
                    lines.append(text[current_pos:break_pos])
                    current_pos = break_pos
                else:
                    # This segment needs to be broken further
                    while current_pos < break_pos:
                        chunk_end = min(current_pos + max_chars_per_line, break_pos)
                        lines.append(text[current_pos:chunk_end])
                        current_pos = chunk_end
                
                # If we've used up all our lines, stop
                if len(lines) >= max_lines:
                    break
            
            # Add any remaining text if we have space
            if current_pos < len(text) and len(lines) < max_lines:
                remaining_length = len(text) - current_pos
                
                if remaining_length <= max_chars_per_line:
                    lines.append(text[current_pos:])
                else:
                    # Need to break the last part
                    lines.append(text[current_pos:current_pos + max_chars_per_line])
                    
                    # Add ellipsis if there's more text
                    if current_pos + max_chars_per_line < len(text) and not lines[-1].endswith('…'):
                        lines[-1] = lines[-1][:-1] + '…'
        else:
            # If no natural breaks, simply chunk by character count
            for i in range(0, len(text), max_chars_per_line):
                if len(lines) >= max_lines:
                    # If we're on the last line and have more text, add ellipsis
                    if i < len(text) and not lines[-1].endswith('…'):
                        lines[-1] = lines[-1][:-1] + '…'
                    break
                
                end = min(i + max_chars_per_line, len(text))
                lines.append(text[i:end])
        
        return lines
    
    def analyze_reading_complexity(self, text: str) -> Dict[str, Any]:
        """
        Analyze text complexity to help with smart breaking decisions.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with analysis metrics
        """
        if not self.nltk_available:
            return {"complexity": "unknown", "avg_word_length": 0, "long_words_ratio": 0}
        
        try:
            words = word_tokenize(text)
            if not words:
                return {"complexity": "unknown", "avg_word_length": 0, "long_words_ratio": 0}
            
            word_lengths = [len(word) for word in words if word.isalnum()]
            avg_word_length = sum(word_lengths) / len(word_lengths) if word_lengths else 0
            
            # Count long words (>6 chars)
            long_words = [w for w in word_lengths if w > 6]
            long_words_ratio = len(long_words) / len(word_lengths) if word_lengths else 0
            
            # Determine complexity
            if avg_word_length > 5.5 or long_words_ratio > 0.3:
                complexity = "high"
            elif avg_word_length > 4.5 or long_words_ratio > 0.2:
                complexity = "medium"
            else:
                complexity = "low"
            
            return {
                "complexity": complexity,
                "avg_word_length": avg_word_length,
                "long_words_ratio": long_words_ratio
            }
        except Exception as e:
            self.logger.warning(f"Text complexity analysis failed: {str(e)}")
            return {"complexity": "unknown", "avg_word_length": 0, "long_words_ratio": 0}
    
    def optimize_line_count(self, text: str, max_lines: int = 2) -> int:
        """
        Determine the optimal number of lines for a text segment.
        
        Args:
            text: Text to analyze
            max_lines: Maximum allowed lines
            
        Returns:
            Recommended number of lines (1 to max_lines)
        """
        # Short text can use one line
        if len(text) < self.min_line_length * 2:
            return 1
        
        # Very long text should use maximum lines
        if len(text) > self.optimal_line_length * max_lines * 0.85:
            return max_lines
        
        # For medium-length text, check complexity
        complexity = self.analyze_reading_complexity(text)
        
        # High complexity text benefits from more lines (easier to read)
        if complexity["complexity"] == "high" and len(text) > self.optimal_line_length * 1.5:
            return min(2, max_lines)
        
        # Medium length text with medium complexity
        if len(text) > self.optimal_line_length * 1.2:
            return min(2, max_lines)
        
        # Default to a single line for better flow
        return 1 