#!/usr/bin/env python3
"""
Emphasis Detection Example

This script demonstrates the automatic emphasis detection functionality
of the Subtitle Generation System, which identifies and formats emphasized text.
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path to import from app
sys.path.append(str(Path(__file__).parent.parent))

from app.services.subtitles import EmphasisDetector, EmphasisFormat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_emphasized_text(title, original_text, detector, subtitle_format='vtt'):
    """Print the original and emphasized text side by side."""
    emphasized_text = detector.apply_emphasis(original_text, subtitle_format)
    
    print(f"\n{title}:")
    print("-" * len(title))
    print(f"Original:   \"{original_text}\"")
    print(f"Emphasized: \"{emphasized_text}\"")

def main():
    """Run the emphasis detection examples."""
    print("Emphasis Detection Examples")
    print("==========================")
    
    # Create a default emphasis detector
    detector = EmphasisDetector()
    
    # Example texts with different types of emphasis
    examples = {
        "Markdown Bold": "This has **bold words** in it.",
        "Markdown Italic": "This has *italic words* in it.",
        "Markdown Combined": "This has *italic* and **bold** formatting.",
        "Uppercase Words": "This has IMPORTANT words in it.",
        "Key Phrases": "Remember this essential fact.",
        "Quoted Text": "He said \"this is important\" to remember."
    }
    
    # Example 1: Different emphasis detection methods
    for title, text in examples.items():
        print_emphasized_text(title, text, detector)
        
    # Example 2: Different subtitle formats
    print("\n\nEMPHASIS IN DIFFERENT SUBTITLE FORMATS")
    print("======================================")
    
    sample_text = "This has **bold words** and *italic words* in it."
    
    for subtitle_format in ['vtt', 'srt', 'ass', 'ttml']:
        print_emphasized_text(
            f"{subtitle_format.upper()} Format", 
            sample_text, 
            detector, 
            subtitle_format
        )
    
    # Example 3: Custom detector configuration
    print("\n\nCUSTOM DETECTOR CONFIGURATIONS")
    print("==============================")
    
    # Create a detector that uses colors for emphasis
    color_detector = EmphasisDetector(config={
        'emphasis_format': EmphasisFormat.COLOR,
        'emphasis_color': '#FF0000',  # Red color
        'detect_quotes': True
    })
    
    print_emphasized_text(
        "Color Emphasis", 
        "This has **colored words** and \"quoted text\" in it.", 
        color_detector
    )
    
    # Create a detector that only uses italic for emphasis
    italic_detector = EmphasisDetector(config={
        'emphasis_format': EmphasisFormat.ITALIC,
        'key_phrase_format': EmphasisFormat.ITALIC,
        'uppercase_format': EmphasisFormat.ITALIC,
        'detect_uppercase': True,
        'detect_key_phrases': True
    })
    
    print_emphasized_text(
        "Italic Only Emphasis", 
        "This has IMPORTANT words and remember this critical fact.", 
        italic_detector
    )
    
    # Example 4: Processing a transcript
    print("\n\nPROCESSING A TRANSCRIPT")
    print("======================")
    
    # Create a simple transcript
    transcript = {
        "segments": [
            {
                "start": 0.0,
                "end": 3.0,
                "text": "This is a **bold** statement."
            },
            {
                "start": 3.5,
                "end": 6.0,
                "text": "And this is an *italic* phrase."
            },
            {
                "start": 6.5,
                "end": 10.0,
                "text": "This is a CRITICAL warning message."
            }
        ]
    }
    
    # Process the transcript
    processed = detector.process_transcript(transcript, 'vtt')
    
    print("Original Transcript:")
    for segment in transcript["segments"]:
        print(f"  {segment['text']}")
    
    print("\nProcessed Transcript:")
    for segment in processed["segments"]:
        print(f"  {segment['text']}")

if __name__ == "__main__":
    main() 