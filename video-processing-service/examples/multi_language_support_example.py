#!/usr/bin/env python3
"""
Multi-Language Support Example

This script demonstrates the multi-language support capabilities of the 
Subtitle Generation System, which enables proper handling of different
writing systems, text directions, and character sets.
"""

import sys
import logging
import asyncio
from pathlib import Path

# Add parent directory to path to import from app
sys.path.append(str(Path(__file__).parent.parent))

from app.services.subtitles import (
    LanguageSupport, TextDirection, LanguageScript,
    SubtitleGenerator, SubtitleFormat, SubtitleStyle
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sample texts in different languages to demonstrate multi-language support
SAMPLE_TEXTS = {
    "en": "This is a sample English subtitle with some emphasis on **important** words.",
    "es": "Este es un subtítulo de ejemplo en español con énfasis en palabras **importantes**.",
    "fr": "Ceci est un exemple de sous-titre en français avec l'accent sur les mots **importants**.",
    "de": "Dies ist ein Beispiel-Untertitel auf Deutsch mit Betonung auf **wichtigen** Wörtern.",
    "zh": "这是一个中文字幕示例，强调一些**重要**文字。",
    "ja": "これは日本語の字幕サンプルで、**重要な**単語に強調があります。",
    "ar": "هذا مثال على الترجمة العربية مع التركيز على الكلمات **المهمة**.",
    "he": "זוהי כתובית לדוגמה בעברית עם דגש על מילים **חשובות**.",
    "ru": "Это пример субтитров на русском языке с акцентом на **важных** словах.",
    "hi": "यह हिंदी में एक नमूना उपशीर्षक है जिसमें **महत्वपूर्ण** शब्दों पर जोर दिया गया है।",
    "ko": "이것은 **중요한** 단어를 강조하는 한국어 자막의 예입니다."
}

async def generate_sample_subtitles(output_dir: str):
    """
    Generate subtitle files in different languages.
    
    Args:
        output_dir: Directory to save subtitle files
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize language support
    language_support = LanguageSupport()
    
    # Initialize subtitle generator with language support
    generator = SubtitleGenerator(
        default_format=SubtitleFormat.VTT,
        config={
            "language_config": {
                "auto_detect_language": False,
                "normalize_unicode": True
            }
        }
    )
    
    # Print supported languages
    print("\nSUPPORTED LANGUAGES")
    print("===================")
    
    supported_languages = sorted([
        (code, language_support.get_language_name(code))
        for code in language_support.LANGUAGE_SCRIPTS.keys()
    ], key=lambda x: x[1])  # Sort by language name
    
    for code, name in supported_languages:
        script = language_support.get_script_for_language(code)
        direction = language_support.get_text_direction(code)
        print(f"{name} ({code}): {script.value} script, {direction.value} direction")
    
    # Generate subtitles for each language
    print("\nGENERATING SUBTITLES IN MULTIPLE LANGUAGES")
    print("=========================================")
    
    for lang_code, text in SAMPLE_TEXTS.items():
        # Create a simple transcript with one segment
        transcript = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": text
                }
            ]
        }
        
        # Get language name
        lang_name = language_support.get_language_name(lang_code)
        
        # Generate subtitles in different formats
        for format_name in ["vtt", "srt", "ass"]:
            format_enum = SubtitleFormat(format_name)
            output_file = output_path / f"subtitle_{lang_code}_{format_name}.{format_name}"
            
            # Generate subtitle file
            await generator.generate_subtitles(
                transcript=transcript,
                output_path=str(output_file),
                format=format_enum,
                language=lang_code,
                auto_detect_language=False,
                detect_emphasis=True
            )
            
            print(f"Generated {format_name.upper()} subtitle in {lang_name} ({lang_code}): {output_file}")

def print_language_examples():
    """Print examples of text formatting in different languages."""
    print("\nLANGUAGE FORMATTING EXAMPLES")
    print("==========================")
    
    language_support = LanguageSupport()
    
    for lang_code, text in SAMPLE_TEXTS.items():
        lang_name = language_support.get_language_name(lang_code)
        print(f"\n{lang_name} ({lang_code}):")
        print("-" * len(lang_name + f" ({lang_code})"))
        
        # Print original text
        print(f"Original:  {text}")
        
        # Print with language-specific formatting for VTT
        vtt_formatted = language_support.format_subtitle_for_language(text, lang_code, 'vtt')
        print(f"VTT:       {vtt_formatted}")
        
        # Print with language-specific formatting for ASS
        ass_formatted = language_support.format_subtitle_for_language(text, lang_code, 'ass')
        print(f"ASS:       {ass_formatted}")
        
        # Print recommended fonts
        fonts = language_support.get_recommended_fonts(lang_code)
        print(f"Fonts:     {', '.join(fonts[:3])}")
        
        # Print text direction
        direction = language_support.get_text_direction(lang_code)
        print(f"Direction: {direction.value}")

def demonstrate_script_detection():
    """Demonstrate automatic language detection."""
    print("\nLANGUAGE DETECTION")
    print("=================")
    
    language_support = LanguageSupport()
    
    # Skip if langdetect is not available
    if not language_support.langdetect_available:
        print("Language detection not available (langdetect not installed)")
        return
    
    for lang_code, text in SAMPLE_TEXTS.items():
        try:
            detected = language_support.detect_language(text)
            lang_name = language_support.get_language_name(lang_code)
            detected_name = language_support.get_language_name(detected)
            
            result = "✓ MATCH" if detected == lang_code else f"✗ MISMATCH (detected as {detected_name})"
            print(f"{lang_name} ({lang_code}): {result}")
        except Exception as e:
            print(f"{lang_code}: Detection failed - {str(e)}")

async def main():
    """Run the multi-language support examples."""
    print("Multi-Language Support Examples")
    print("===============================")
    
    # Demonstrate text formatting
    print_language_examples()
    
    # Demonstrate language detection
    demonstrate_script_detection()
    
    # Generate sample subtitle files
    await generate_sample_subtitles("./output/multi_language_examples")

if __name__ == "__main__":
    asyncio.run(main()) 