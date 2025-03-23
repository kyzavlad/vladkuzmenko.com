#!/usr/bin/env python3
"""
Script Timing Preservation Example

This example demonstrates how to preserve timing information when translating
video scripts for dubbing or subtitling. It shows how to:

1. Load a script with timed segments
2. Translate each segment while preserving terminology
3. Optimize timing for the translated segments
4. Check for potential timing issues
5. Generate an output script with adjusted timing

Usage: python -m app.avatar_creation.video_translation.examples.timing_example
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any

from app.avatar_creation.video_translation.translator import (
    NeuralTranslator, 
    ContextAwareTranslator,
    TranslationOptions
)
from app.avatar_creation.video_translation.terminology import (
    TerminologyManager,
    TermDefinition
)
from app.avatar_creation.video_translation.timing import (
    ScriptTimingPreserver,
    TimedSegment
)


def load_example_script() -> Dict[str, Any]:
    """
    Load an example video script with timing information.
    
    In a real implementation, this would load from a file.
    Here we create a sample script programmatically.
    
    Returns:
        A dictionary representing a script with timed segments
    """
    # Create a sample product demo script
    script = {
        "title": "Avatar Creation System Demo",
        "language": "en",
        "duration": 60.0,  # seconds
        "segments": [
            {
                "id": "intro_1",
                "text": "Welcome to the Avatar Creation System demonstration.",
                "start_time": 0.0,
                "end_time": 3.0
            },
            {
                "id": "intro_2",
                "text": "Today, we'll show you how to create lifelike avatars with our revolutionary neural network technology.",
                "start_time": 3.5,
                "end_time": 8.5
            },
            {
                "id": "feature_1",
                "text": "First, let's explore the facial modeling capabilities. Our system captures over 50 facial landmarks.",
                "start_time": 9.0,
                "end_time": 14.0
            },
            {
                "id": "feature_2",
                "text": "Next, we'll demonstrate the voice cloning technology. Just a 5-second sample is enough to create a matching voice.",
                "start_time": 14.5,
                "end_time": 20.5
            },
            {
                "id": "feature_3",
                "text": "The gesture recognition system allows for natural body language and movement patterns.",
                "start_time": 21.0,
                "end_time": 25.0
            },
            {
                "id": "detail_1",
                "text": "Our proprietary deep learning algorithms ensure the highest quality results in the industry.",
                "start_time": 25.5,
                "end_time": 30.5
            },
            {
                "id": "detail_2",
                "text": "The system processes everything in real-time, allowing for immediate feedback and adjustments.",
                "start_time": 31.0,
                "end_time": 36.0
            },
            {
                "id": "conclusion_1",
                "text": "With the Avatar Creation System, you can create digital versions of yourself for virtual meetings, presentations, or entertainment.",
                "start_time": 36.5,
                "end_time": 43.5
            },
            {
                "id": "conclusion_2",
                "text": "Contact us today to schedule a personalized demonstration of this groundbreaking technology.",
                "start_time": 44.0,
                "end_time": 49.0
            }
        ],
        "metadata": {
            "version": "1.0",
            "speaker": "Presenter",
            "created_at": "2023-07-15"
        }
    }
    
    return script


def setup_terminology_manager() -> TerminologyManager:
    """
    Set up a terminology manager with relevant technical terms.
    
    Returns:
        Configured TerminologyManager
    """
    term_manager = TerminologyManager()
    
    # Add technical terms with their translations
    term_manager.add_term(TermDefinition(
        source_term="Avatar Creation System",
        translations={
            "es": "Sistema de Creación de Avatares",
            "fr": "Système de Création d'Avatars",
            "de": "Avatar-Erstellungssystem",
            "ja": "アバター作成システム",
            "zh": "虚拟形象创建系统"
        },
        domain="product",
        description="The product name",
        part_of_speech="noun"
    ))
    
    term_manager.add_term(TermDefinition(
        source_term="neural network",
        translations={
            "es": "red neuronal",
            "fr": "réseau de neurones",
            "de": "neuronales Netzwerk",
            "ja": "ニューラルネットワーク",
            "zh": "神经网络"
        },
        domain="ai",
        description="A computer system modeled on the human brain",
        part_of_speech="noun"
    ))
    
    term_manager.add_term(TermDefinition(
        source_term="facial landmarks",
        translations={
            "es": "puntos de referencia faciales",
            "fr": "points de repère faciaux",
            "de": "Gesichtsreferenzpunkte",
            "ja": "顔のランドマーク",
            "zh": "面部特征点"
        },
        domain="computer_vision",
        description="Specific points on a face used for analysis",
        part_of_speech="noun"
    ))
    
    term_manager.add_term(TermDefinition(
        source_term="voice cloning",
        translations={
            "es": "clonación de voz",
            "fr": "clonage vocal",
            "de": "Stimmklonen",
            "ja": "音声複製",
            "zh": "声音克隆"
        },
        domain="audio",
        description="Technology to replicate someone's voice",
        part_of_speech="noun"
    ))
    
    term_manager.add_term(TermDefinition(
        source_term="deep learning",
        translations={
            "es": "aprendizaje profundo",
            "fr": "apprentissage profond",
            "de": "tiefes Lernen",
            "ja": "深層学習",
            "zh": "深度学习"
        },
        domain="ai",
        description="A subset of machine learning based on neural networks",
        part_of_speech="noun"
    ))
    
    return term_manager


def translate_script(script: Dict[str, Any], source_lang: str, target_lang: str,
                    term_manager: TerminologyManager) -> Dict[str, Dict[str, str]]:
    """
    Translate a script from source language to target language.
    
    Args:
        script: The script to translate
        source_lang: Source language code
        target_lang: Target language code
        term_manager: Terminology manager for preserving technical terms
        
    Returns:
        Dictionary mapping segment IDs to translations
    """
    print(f"Translating script from {source_lang} to {target_lang}...")
    
    # Initialize the translator
    translator = ContextAwareTranslator()
    
    # Configure translation options
    options = TranslationOptions(
        preserve_technical_terms=True,
        preserve_named_entities=True
    )
    
    # Collect translations
    translations = {target_lang: {}}
    
    for segment in script["segments"]:
        segment_id = segment["id"]
        text = segment["text"]
        
        # Process text for terminology
        processed_text, terms = term_manager.process_text(text, source_lang, target_lang)
        
        # Translate the text
        translated_text, metadata = translator.translate_text(
            text, source_lang, target_lang, options
        )
        
        # In a real implementation, the translator would use the terminology manager directly.
        # For this example, we'll simulate replacing terms in the translated text.
        for term in terms:
            term_text = term["term"]
            term_translation = term.get("translation")
            
            if term_translation and term_text.lower() in text.lower():
                # Simple replacement - in a real system this would be more sophisticated
                # to handle case sensitivity, word boundaries, etc.
                translated_text = translated_text.replace(
                    f"[{target_lang}] {term_text}",
                    term_translation
                )
        
        # Store the translation
        translations[target_lang][segment_id] = translated_text
        
        print(f"  - Segment {segment_id}: Translated")
    
    return translations


def adjust_script_timing(script: Dict[str, Any], translations: Dict[str, Dict[str, str]],
                        source_lang: str, target_lang: str) -> Dict[str, Any]:
    """
    Adjust timing information in the script for the translated content.
    
    Args:
        script: The original script
        translations: Dictionary mapping segment IDs to translations
        source_lang: Source language code
        target_lang: Target language code
        
    Returns:
        A new script with adjusted timing for the translated content
    """
    print(f"Adjusting timing for {target_lang} translations...")
    
    # Initialize the timing preserver
    timing_preserver = ScriptTimingPreserver()
    
    # Process the script
    translated_script = timing_preserver.process_script(
        script, translations, source_lang, target_lang
    )
    
    # Update metadata
    translated_script["language"] = target_lang
    if "metadata" not in translated_script:
        translated_script["metadata"] = {}
    translated_script["metadata"]["translated_from"] = source_lang
    translated_script["metadata"]["timing_adjusted"] = True
    
    return translated_script


def save_script(script: Dict[str, Any], filename: str) -> None:
    """
    Save a script to a JSON file.
    
    Args:
        script: The script to save
        filename: The filename to save to
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save the script
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(script, f, indent=2, ensure_ascii=False)
    
    print(f"Script saved to {filename}")


def analyze_timing_issues(script: Dict[str, Any], language: str) -> None:
    """
    Analyze timing issues in a script.
    
    Args:
        script: The script to analyze
        language: The language code
    """
    print(f"Analyzing timing issues for {language} script...")
    
    timing_preserver = ScriptTimingPreserver()
    
    issues_found = False
    
    for segment_data in script["segments"]:
        segment = TimedSegment(
            text=segment_data["text"],
            start_time=segment_data["start_time"],
            end_time=segment_data["end_time"],
            segment_id=segment_data["id"]
        )
        
        issues = timing_preserver.check_timing_issues(segment, language)
        
        if issues["has_issues"]:
            issues_found = True
            print(f"  - Segment {segment.segment_id}: Issues detected")
            
            if issues["too_fast"]:
                print(f"    * Speaking rate too fast: {issues['chars_per_second']:.2f} chars/sec")
                print(f"    * Recommended duration: {issues['recommended_duration']:.2f}s")
            
            if issues.get("too_slow", False):
                print(f"    * Speaking rate too slow: {issues['chars_per_second']:.2f} chars/sec")
    
    if not issues_found:
        print("  - No timing issues found")


def print_script_comparison(original: Dict[str, Any], translated: Dict[str, Any]) -> None:
    """
    Print a comparison of original and translated scripts.
    
    Args:
        original: The original script
        translated: The translated script
    """
    print("\n=== Script Comparison ===")
    print(f"Original language: {original['language']}")
    print(f"Translated language: {translated['language']}")
    print("\nSegments:")
    
    # Match segments by ID
    for orig_segment in original["segments"]:
        seg_id = orig_segment["id"]
        
        # Find corresponding translated segment
        trans_segment = next(
            (s for s in translated["segments"] if s["id"] == seg_id),
            None
        )
        
        if trans_segment:
            print(f"\nSegment: {seg_id}")
            print(f"  Original ({original['language']}): {orig_segment['text']}")
            print(f"    Time: {orig_segment['start_time']:.1f}s - {orig_segment['end_time']:.1f}s, "
                  f"Duration: {orig_segment['end_time'] - orig_segment['start_time']:.1f}s")
            
            print(f"  Translated ({translated['language']}): {trans_segment['text']}")
            print(f"    Time: {trans_segment['start_time']:.1f}s - {trans_segment['end_time']:.1f}s, "
                  f"Duration: {trans_segment['end_time'] - trans_segment['start_time']:.1f}s")
            
            # Calculate character rates
            orig_chars_per_sec = len(orig_segment['text']) / (orig_segment['end_time'] - orig_segment['start_time'])
            trans_chars_per_sec = len(trans_segment['text']) / (trans_segment['end_time'] - trans_segment['start_time'])
            
            print(f"  Chars/sec - Original: {orig_chars_per_sec:.1f}, Translated: {trans_chars_per_sec:.1f}")


def main():
    """Main function to run the timing preservation example."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Script Timing Preservation Example"
    )
    
    parser.add_argument("--source-lang", type=str, default="en",
                       help="Source language code")
    parser.add_argument("--target-lang", type=str, default="es",
                       help="Target language code")
    parser.add_argument("--output-dir", type=str, default="output/translations",
                       help="Output directory for translated scripts")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Script Timing Preservation Example")
    print("=" * 80)
    
    # Step 1: Load the example script
    script = load_example_script()
    print(f"Loaded example script with {len(script['segments'])} segments")
    
    # Step 2: Set up terminology manager
    term_manager = setup_terminology_manager()
    print(f"Set up terminology manager with {len(term_manager.terms)} terms")
    
    # Step 3: Translate the script
    translations = translate_script(
        script, args.source_lang, args.target_lang, term_manager
    )
    
    # Step 4: Adjust timing for translated script
    translated_script = adjust_script_timing(
        script, translations, args.source_lang, args.target_lang
    )
    
    # Step 5: Save the original and translated scripts
    original_filename = os.path.join(args.output_dir, f"script_{args.source_lang}.json")
    translated_filename = os.path.join(args.output_dir, f"script_{args.target_lang}.json")
    
    save_script(script, original_filename)
    save_script(translated_script, translated_filename)
    
    # Step 6: Analyze for timing issues
    analyze_timing_issues(translated_script, args.target_lang)
    
    # Step 7: Print comparison
    print_script_comparison(script, translated_script)
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main() 