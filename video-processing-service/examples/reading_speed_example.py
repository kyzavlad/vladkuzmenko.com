#!/usr/bin/env python3
"""
Reading Speed Calculator Example

This script demonstrates the functionality of the ReadingSpeedCalculator class
for calibrating subtitle durations based on text content and audience reading speed.
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path to import from app
sys.path.append(str(Path(__file__).parent.parent))

from app.services.subtitles import ReadingSpeedCalculator, AudienceType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def format_duration(seconds):
    """Format seconds as minutes and seconds."""
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes}:{seconds:05.2f}"

def print_subtitle_duration(title, text, calculator):
    """Print the calculated duration for a subtitle."""
    duration = calculator.calculate_duration(text)
    print(f"\n{title}:")
    print("-" * len(title))
    print(f"Text: \"{text}\"")
    print(f"Duration: {format_duration(duration)} ({duration:.2f} seconds)")
    
    # Calculate characters per second
    char_count = len(text.replace(" ", ""))
    cps = char_count / duration if duration > 0 else 0
    print(f"Characters: {char_count}, Characters per second: {cps:.2f}")
    
    # Calculate words per second
    word_count = len(text.split())
    wps = word_count / duration if duration > 0 else 0
    print(f"Words: {word_count}, Words per second: {wps:.2f}")

def main():
    """Run the reading speed calculator examples."""
    print("Reading Speed Calculator Examples")
    print("================================")
    
    # Create calculators for different audience types
    calculators = {
        "Children": ReadingSpeedCalculator(config={"audience_type": AudienceType.CHILDREN}),
        "General": ReadingSpeedCalculator(config={"audience_type": AudienceType.GENERAL}),
        "Experienced": ReadingSpeedCalculator(config={"audience_type": AudienceType.EXPERIENCED}),
        "Speed Reader": ReadingSpeedCalculator(config={"audience_type": AudienceType.SPEED_READER})
    }
    
    # Sample texts with different characteristics
    texts = {
        "Short simple": "This is a short sentence.",
        "Medium length": "This is a medium length sentence that would be typical for a subtitle in a documentary or film.",
        "Technical": "The HTTP 1.1 protocol uses TCP port 443 for HTTPS connections with a 128-bit encryption key.",
        "Complex vocabulary": "The juxtaposition of disparate lexical elements facilitates the comprehensive assimilation of multifaceted concepts.",
        "Long with multiple sentences": "This is a longer piece of text. It contains multiple sentences with varying lengths. These would typically be split into multiple subtitles, but this example shows the duration calculation for the full text."
    }
    
    # Example 1: Compare different audience types with the same text
    print("\nEXAMPLE 1: COMPARISON OF AUDIENCE TYPES")
    print("=======================================")
    
    sample_text = texts["Medium length"]
    print(f"Sample text: \"{sample_text}\"")
    
    for audience_name, calculator in calculators.items():
        duration = calculator.calculate_duration(sample_text)
        speeds = calculator.get_reading_speeds()
        print(f"{audience_name}: {format_duration(duration)} ({duration:.2f} seconds)")
        print(f"  Reading speeds: {speeds['wpm']} WPM, {speeds['cpm']} CPM")
    
    # Example 2: Compare different text types with the same audience
    print("\nEXAMPLE 2: COMPARISON OF TEXT TYPES")
    print("==================================")
    
    general_calculator = calculators["General"]
    
    for text_type, text in texts.items():
        print_subtitle_duration(text_type, text, general_calculator)
    
    # Example 3: Compare different calculation methods
    print("\nEXAMPLE 3: COMPARISON OF CALCULATION METHODS")
    print("===========================================")
    
    sample_text = texts["Medium length"]
    print(f"Sample text: \"{sample_text}\"")
    
    methods = {
        "Character-based": ReadingSpeedCalculator(config={"calculation_method": "character"}),
        "Word-based": ReadingSpeedCalculator(config={"calculation_method": "word"}),
        "Syllable-based": ReadingSpeedCalculator(config={"calculation_method": "syllable"})
    }
    
    for method_name, calculator in methods.items():
        duration = calculator.calculate_duration(sample_text)
        print(f"{method_name}: {format_duration(duration)} ({duration:.2f} seconds)")
    
    # Example 4: Language-specific adjustments
    print("\nEXAMPLE 4: LANGUAGE-SPECIFIC ADJUSTMENTS")
    print("=======================================")
    
    languages = {
        "English": "en",
        "German": "de", 
        "French": "fr",
        "Spanish": "es",
        "Japanese": "ja",
        "Chinese": "zh"
    }
    
    sample_text = texts["Medium length"]
    print(f"Sample text: \"{sample_text}\"")
    
    for language_name, language_code in languages.items():
        calculator = ReadingSpeedCalculator(config={"language": language_code})
        duration = calculator.calculate_duration(sample_text)
        adjustment = calculator.language_factor
        print(f"{language_name} ({language_code}): {format_duration(duration)} ({duration:.2f} seconds)")
        print(f"  Language adjustment factor: {adjustment:.2f}")
    
    # Example 5: Batch processing of subtitles
    print("\nEXAMPLE 5: BATCH SUBTITLE CALIBRATION")
    print("====================================")
    
    subtitles = [
        {"text": "This is the first subtitle.", "start": 0.0, "end": 4.0},
        {"text": "This is the second subtitle, which is a bit longer.", "start": 4.5, "end": 9.0},
        {"text": "Third subtitle with technical terms: HTTP, TCP/IP, GPU.", "start": 10.0, "end": 14.0},
        {"text": "Fourth subtitle that appears after a pause.", "start": 16.0, "end": 20.0}
    ]
    
    calculator = ReadingSpeedCalculator()
    calibrated = calculator.calibrate_subtitle_durations(subtitles)
    
    print("Original vs. Calibrated Durations:")
    print("---------------------------------")
    
    for i, (original, calibrated_sub) in enumerate(zip(subtitles, calibrated)):
        original_duration = original["end"] - original["start"]
        calibrated_duration = calibrated_sub["end"] - calibrated_sub["start"]
        
        print(f"Subtitle {i+1}: \"{original['text']}\"")
        print(f"  Original: {format_duration(original_duration)} ({original_duration:.2f} seconds)")
        print(f"  Calibrated: {format_duration(calibrated_duration)} ({calibrated_duration:.2f} seconds)")
        
        if calibrated_duration > original_duration:
            percent = ((calibrated_duration / original_duration) - 1) * 100
            print(f"  Increased by: {percent:.1f}%")
        else:
            percent = ((original_duration / calibrated_duration) - 1) * 100
            print(f"  Decreased by: {percent:.1f}%")

if __name__ == "__main__":
    main() 