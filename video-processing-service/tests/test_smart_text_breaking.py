import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.subtitles import SmartTextBreaker


class TestSmartTextBreaking(unittest.TestCase):
    """Tests for the SmartTextBreaker class."""

    def setUp(self):
        """Set up test environment."""
        # Create a basic text breaker
        self.basic_breaker = SmartTextBreaker(config={"breaking_strategy": "basic"})
        
        # Create a breaker with linguistic strategy
        # Mock NLTK availability
        with patch.object(SmartTextBreaker, '_initialize_nltk', return_value=True):
            self.linguistic_breaker = SmartTextBreaker(
                config={"breaking_strategy": "linguistic"}
            )
        
        # Create a breaker with balanced strategy
        self.balanced_breaker = SmartTextBreaker(config={"breaking_strategy": "balanced"})
        
        # Create a breaker for languages without spaces
        self.cjk_breaker = SmartTextBreaker(config={"language": "zh"})
        
        # Sample texts for testing
        self.simple_text = "This is a simple text for testing."
        self.complex_text = ("This is a more complex text with multiple sentences. "
                             "It should be broken at natural points. Linguistic analysis should help.")
        self.unbalanced_text = "Short first part. But this second part is significantly longer and needs better balancing."
        self.cjk_text = "这是一个中文测试文本，用来测试没有空格的语言。"

    def test_basic_breaking(self):
        """Test basic space-based text breaking."""
        # Test with simple text
        lines = self.basic_breaker.break_text(self.simple_text, max_chars_per_line=20)
        self.assertGreater(len(lines), 0)
        self.assertLessEqual(max(len(line) for line in lines), 20)
        
        # Test with complex text and max lines
        lines = self.basic_breaker.break_text(self.complex_text, max_chars_per_line=30, max_lines=2)
        self.assertEqual(len(lines), 2)
        self.assertLessEqual(max(len(line) for line in lines), 30)
        
        # Test empty text
        lines = self.basic_breaker.break_text("", max_chars_per_line=30)
        self.assertEqual(lines, [])
        
        # Test text that fits in one line
        short_text = "Short text."
        lines = self.basic_breaker.break_text(short_text, max_chars_per_line=30)
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0], short_text)

    @patch('nltk.tokenize.sent_tokenize')
    def test_linguistic_breaking(self, mock_sent_tokenize):
        """Test linguistic text breaking with mocked NLTK."""
        # Mock sentence tokenization
        mock_sent_tokenize.return_value = [
            "This is the first sentence.",
            "This is the second sentence."
        ]
        
        # Test breaking with mocked tokenization
        lines = self.linguistic_breaker.break_text(self.complex_text, max_chars_per_line=30)
        self.assertEqual(len(lines), 2)
        mock_sent_tokenize.assert_called_once()

    def test_balanced_breaking(self):
        """Test balanced text breaking."""
        # Test with unbalanced text
        lines = self.balanced_breaker.break_text(self.unbalanced_text, max_chars_per_line=40)
        self.assertEqual(len(lines), 2)
        
        # Check that lines are more balanced than with basic breaking
        basic_lines = self.basic_breaker.break_text(self.unbalanced_text, max_chars_per_line=40)
        
        # Calculate length difference in basic breaking
        basic_diff = abs(len(basic_lines[0]) - len(basic_lines[1]))
        
        # Calculate length difference in balanced breaking
        balanced_diff = abs(len(lines[0]) - len(lines[1]))
        
        # Balanced breaking should produce more even line lengths
        self.assertLessEqual(balanced_diff, basic_diff)

    def test_cjk_breaking(self):
        """Test breaking for languages without spaces (CJK)."""
        lines = self.cjk_breaker.break_text(self.cjk_text, max_chars_per_line=10, max_lines=3)
        self.assertLessEqual(len(lines), 3)
        self.assertLessEqual(max(len(line) for line in lines), 10)

    def test_find_best_break_point(self):
        """Test finding the best break point in text."""
        test_text = "This is a test sentence with commas, periods. And multiple parts."
        
        # Test breaking at period
        break_point = self.linguistic_breaker._find_best_break_point(test_text, 30)
        self.assertEqual(test_text[break_point-2:break_point], ". ")
        
        # Test breaking at comma when max_chars is lower
        break_point = self.linguistic_breaker._find_best_break_point(test_text, 20)
        self.assertEqual(test_text[break_point-2:break_point], ", ")
        
        # Test breaking at space when no better option
        break_point = self.linguistic_breaker._find_best_break_point("ThisIsATestWithNoGoodBreaks but spaces", 20)
        self.assertEqual(break_point, 21)  # Should break after 'ThisIsATestWithNoGoodBreaks'

    @patch('nltk.tokenize.word_tokenize')
    def test_analyze_reading_complexity(self, mock_word_tokenize):
        """Test analysis of text complexity."""
        # Mock word tokenization
        mock_word_tokenize.return_value = ["This", "is", "a", "complex", "test"]
        
        # Test complexity analysis
        result = self.linguistic_breaker.analyze_reading_complexity("This is a complex test")
        self.assertIn("complexity", result)
        self.assertIn("avg_word_length", result)
        mock_word_tokenize.assert_called_once()
        
        # Test with no NLTK
        no_nltk_breaker = SmartTextBreaker(config={"breaking_strategy": "basic"})
        no_nltk_breaker.nltk_available = False
        result = no_nltk_breaker.analyze_reading_complexity("Test text")
        self.assertEqual(result["complexity"], "unknown")

    def test_optimize_line_count(self):
        """Test optimizing the number of lines for text."""
        # Test with short text (should use 1 line)
        lines = self.linguistic_breaker.optimize_line_count("Short text.")
        self.assertEqual(lines, 1)
        
        # Test with long text (should use max allowed)
        very_long_text = "This is a very long piece of text that should definitely use multiple lines " * 5
        lines = self.linguistic_breaker.optimize_line_count(very_long_text, max_lines=3)
        self.assertEqual(lines, 3)
        
        # Test with medium text
        medium_text = "This is medium length text that could be one or two lines."
        lines = self.linguistic_breaker.optimize_line_count(medium_text, max_lines=2)
        self.assertIn(lines, [1, 2])  # Depending on complexity analysis


if __name__ == "__main__":
    unittest.main() 