import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.subtitles.reading_speed import ReadingSpeedCalculator, AudienceType


class TestReadingSpeedCalculator(unittest.TestCase):
    """Tests for the ReadingSpeedCalculator class."""

    def setUp(self):
        """Set up test environment."""
        # Create calculators with different configurations
        self.default_calculator = ReadingSpeedCalculator()
        
        self.children_calculator = ReadingSpeedCalculator(config={
            'audience_type': AudienceType.CHILDREN
        })
        
        self.speed_reader_calculator = ReadingSpeedCalculator(config={
            'audience_type': AudienceType.SPEED_READER
        })
        
        self.word_based_calculator = ReadingSpeedCalculator(config={
            'calculation_method': 'word'
        })
        
        self.syllable_based_calculator = ReadingSpeedCalculator(config={
            'calculation_method': 'syllable'
        })
        
        self.german_calculator = ReadingSpeedCalculator(config={
            'language': 'de'
        })
        
        # Sample texts for testing
        self.short_text = "Short text."
        self.medium_text = "This is a medium length text for testing subtitle duration."
        self.long_text = "This is a longer piece of text that contains multiple sentences. It should result in a longer duration. We need to ensure that the calculator handles this appropriately."
        self.technical_text = "The HTTP 1.1 protocol uses TCP port 443 for HTTPS connections with a 128-bit encryption key."
        self.german_text = "Donaudampfschifffahrtsgesellschaftskapitän ist ein sehr langes deutsches Wort."

    def test_basic_duration_calculation(self):
        """Test basic duration calculation for different text lengths."""
        # Test with short text
        short_duration = self.default_calculator.calculate_duration(self.short_text)
        self.assertGreaterEqual(short_duration, self.default_calculator.min_duration)
        
        # Test with medium text
        medium_duration = self.default_calculator.calculate_duration(self.medium_text)
        self.assertGreater(medium_duration, short_duration)
        
        # Test with long text
        long_duration = self.default_calculator.calculate_duration(self.long_text)
        self.assertGreater(long_duration, medium_duration)
        
        # Test empty text (should return minimum duration)
        empty_duration = self.default_calculator.calculate_duration("")
        self.assertEqual(empty_duration, self.default_calculator.min_duration)

    def test_audience_type_impact(self):
        """Test that different audience types result in different durations."""
        # Children should get longer durations (slower reading)
        children_duration = self.children_calculator.calculate_duration(self.medium_text)
        
        # Default is for general audience (medium speed)
        general_duration = self.default_calculator.calculate_duration(self.medium_text)
        
        # Speed readers get shorter durations (faster reading)
        speed_reader_duration = self.speed_reader_calculator.calculate_duration(self.medium_text)
        
        # Check duration relationships
        self.assertGreater(children_duration, general_duration)
        self.assertGreater(general_duration, speed_reader_duration)

    def test_calculation_methods(self):
        """Test different calculation methods."""
        # Test character-based (default)
        char_duration = self.default_calculator.calculate_duration(self.medium_text)
        
        # Test word-based
        word_duration = self.word_based_calculator.calculate_duration(self.medium_text)
        
        # Test syllable-based
        syllable_duration = self.syllable_based_calculator.calculate_duration(self.medium_text)
        
        # All methods should produce reasonable durations
        for duration in [char_duration, word_duration, syllable_duration]:
            self.assertGreaterEqual(duration, self.default_calculator.min_duration)
            self.assertLessEqual(duration, self.default_calculator.max_duration)

    def test_language_adjustments(self):
        """Test language-specific adjustments."""
        # English text with English calculator (baseline)
        english_duration = self.default_calculator.calculate_duration(self.medium_text)
        
        # English text with German calculator (should be longer due to adjustment factor)
        german_adjusted_duration = self.german_calculator.calculate_duration(self.medium_text)
        
        # German text with German calculator
        german_text_duration = self.german_calculator.calculate_duration(self.german_text)
        
        # Check language adjustment effect
        self.assertGreater(german_adjusted_duration, english_duration)
        
        # German text should have appropriate duration
        self.assertGreaterEqual(german_text_duration, self.german_calculator.min_duration)
        self.assertLessEqual(german_text_duration, self.german_calculator.max_duration)

    def test_special_content_handling(self):
        """Test handling of special content that requires more reading time."""
        # Normal text
        normal_duration = self.default_calculator.calculate_duration(self.medium_text)
        
        # Technical text with numbers, abbreviations
        technical_duration = self.default_calculator.calculate_duration(self.technical_text)
        
        # Technical text should have longer duration per character
        normal_char_count = len(self.medium_text.replace(" ", ""))
        technical_char_count = len(self.technical_text.replace(" ", ""))
        
        # Calculate time per character
        normal_time_per_char = normal_duration / normal_char_count
        technical_time_per_char = technical_duration / technical_char_count
        
        # Technical should take more time per character
        self.assertGreater(technical_time_per_char, normal_time_per_char)
        
        # Test with special content adjustment disabled
        calculator_no_special = ReadingSpeedCalculator(config={'special_content_adjustment': False})
        technical_no_special = calculator_no_special.calculate_duration(self.technical_text)
        
        # Duration should be shorter without special content adjustment
        self.assertLess(technical_no_special, technical_duration)

    def test_duration_constraints(self):
        """Test minimum and maximum duration constraints."""
        # Create calculator with custom constraints
        custom_constraints = ReadingSpeedCalculator(config={
            'min_duration': 1.5,
            'max_duration': 5.0
        })
        
        # Very short text should get minimum duration
        very_short = "Hi."
        short_duration = custom_constraints.calculate_duration(very_short)
        self.assertEqual(short_duration, 1.5)
        
        # Very long text should get maximum duration
        very_long = "This is an extremely long text. " * 20
        long_duration = custom_constraints.calculate_duration(very_long)
        self.assertEqual(long_duration, 5.0)

    def test_subtitle_batch_calibration(self):
        """Test calibration of a batch of subtitles."""
        # Create sample subtitles
        subtitles = [
            {'text': 'Short subtitle.', 'start': 0.0, 'end': 4.0},
            {'text': 'This is a medium length subtitle.', 'start': 4.5, 'end': 8.0},
            {'text': 'This is a longer subtitle with more content to read.', 'start': 9.0, 'end': 13.0}
        ]
        
        # Calibrate durations
        calibrated = self.default_calculator.calibrate_subtitle_durations(subtitles)
        
        # Check results
        self.assertEqual(len(calibrated), len(subtitles))
        
        # Each subtitle should have appropriate duration
        for sub in calibrated:
            duration = sub['end'] - sub['start']
            self.assertGreaterEqual(duration, self.default_calculator.min_duration)
            self.assertLessEqual(duration, self.default_calculator.max_duration)
            self.assertEqual(duration, sub['duration'])
        
        # Check for subtitle collisions
        for i in range(len(calibrated) - 1):
            self.assertLessEqual(calibrated[i]['end'], calibrated[i+1]['start'])

    @patch('pyphen.Pyphen')
    def test_syllable_counting(self, mock_pyphen):
        """Test syllable counting with mocked Pyphen."""
        # Mock pyphen for deterministic testing
        mock_pyphen_instance = MagicMock()
        mock_pyphen_instance.positions.side_effect = lambda word: [2] if len(word) > 3 else []
        mock_pyphen.return_value = mock_pyphen_instance
        
        # Create calculator with syllable method
        calculator = ReadingSpeedCalculator(config={'calculation_method': 'syllable'})
        
        # Test syllable counting
        syllables = calculator._count_syllables("This is a test")
        self.assertGreater(syllables, 0)
        
        # Test duration calculation with syllable method
        duration = calculator.calculate_duration("This is a syllable counting test")
        self.assertGreaterEqual(duration, calculator.min_duration)

    def test_audience_type_setting(self):
        """Test changing audience type."""
        calculator = ReadingSpeedCalculator()
        original_wpm = calculator.wpm
        
        # Change to children
        calculator.set_audience_type(AudienceType.CHILDREN)
        self.assertEqual(calculator.audience_type, AudienceType.CHILDREN)
        self.assertEqual(calculator.wpm, calculator.DEFAULT_WPM[AudienceType.CHILDREN])
        self.assertLess(calculator.wpm, original_wpm)
        
        # Change to speed reader
        calculator.set_audience_type(AudienceType.SPEED_READER)
        self.assertEqual(calculator.audience_type, AudienceType.SPEED_READER)
        self.assertEqual(calculator.wpm, calculator.DEFAULT_WPM[AudienceType.SPEED_READER])
        self.assertGreater(calculator.wpm, original_wpm)


if __name__ == "__main__":
    unittest.main() 