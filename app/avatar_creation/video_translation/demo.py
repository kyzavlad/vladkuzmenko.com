#!/usr/bin/env python3
"""
Video Translation Module Demo

This script demonstrates the capabilities of the Video Translation Module, including:
- Multi-language processing with 50+ language support
- Context-aware neural machine translation
- Industry-specific terminology preservation
- Cultural adaptation for references
- Idiomatic expression handling
- Script timing preservation across languages
- Named entity recognition and preservation
- Technical vocabulary handling

Usage: python -m app.avatar_creation.video_translation.demo
"""

import os
import sys
import argparse
import json
from typing import Dict, List, Any

from app.avatar_creation.video_translation.translator import (
    NeuralTranslator, ContextAwareTranslator, TranslationOptions
)
from app.avatar_creation.video_translation.language_manager import (
    LanguageManager, LanguageProfile, DialectVariant
)
from app.avatar_creation.video_translation.terminology import (
    TerminologyManager, IndustryTerminology,
    NamedEntityRecognizer, TechnicalVocabulary
)
from app.avatar_creation.video_translation.cultural_adaptation import (
    CulturalAdapter, IdiomaticExpressionHandler
)
from app.avatar_creation.video_translation.timing import (
    ScriptTimingPreserver, TimedSegment
)


def create_data_directories():
    """Create necessary directories for data storage."""
    directories = [
        "data/terminology",
        "data/cultural_references",
        "data/idiomatic_expressions",
        "data/vocabulary",
        "config/languages",
        "output/translations"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def demo_basic_translation(args):
    """
    Demonstrate basic translation capabilities.
    
    Args:
        args: Command line arguments
    """
    print("\n=== Basic Translation Demo ===\n")
    
    # Initialize the neural translator
    translator = NeuralTranslator()
    
    # Sample text for translation
    texts = [
        "Welcome to the video translation module demonstration.",
        "This system can translate content into over 50 languages.",
        "It preserves industry terminology and adapts cultural references."
    ]
    
    # Translate to target language
    print(f"Translating from {args.source_lang} to {args.target_lang}:\n")
    
    for text in texts:
        translated, metadata = translator.translate_text(
            text, args.source_lang, args.target_lang
        )
        
        print(f"Original: {text}")
        print(f"Translated: {translated}")
        print(f"Confidence: {metadata['confidence_score']:.2f}")
        print()


def demo_context_aware_translation(args):
    """
    Demonstrate context-aware translation.
    
    Args:
        args: Command line arguments
    """
    print("\n=== Context-Aware Translation Demo ===\n")
    
    # Initialize the context-aware translator
    context_translator = ContextAwareTranslator(context_window=3)
    
    # Sample text with ambiguous terms that need context
    document = [
        "The bank is closed today.",  # 'bank' could be financial institution or river bank
        "People were hoping to withdraw some money.",  # This clarifies 'bank' meaning
        "The spring is running dry.",  # 'spring' could be season or water source
        "We need to find another water source soon."  # This clarifies 'spring' meaning
    ]
    
    # First translate without context
    print(f"Translating WITHOUT context ({args.source_lang} to {args.target_lang}):\n")
    
    for text in document:
        # Translate each sentence individually without context
        translated, metadata = context_translator.translate_text(
            text, args.source_lang, args.target_lang
        )
        
        print(f"Original: {text}")
        print(f"Translated: {translated}")
        print()
    
    # Now translate with context
    print(f"\nTranslating WITH context ({args.source_lang} to {args.target_lang}):\n")
    
    results = context_translator.translate_document(
        document, args.source_lang, args.target_lang
    )
    
    for i, (translated, metadata) in enumerate(results):
        print(f"Original: {document[i]}")
        print(f"Translated: {translated}")
        print(f"Context used: {metadata['context_used']}, Size: {metadata['context_size']}")
        print()


def demo_terminology_preservation(args):
    """
    Demonstrate terminology preservation.
    
    Args:
        args: Command line arguments
    """
    print("\n=== Terminology Preservation Demo ===\n")
    
    # Initialize the terminology manager
    term_manager = TerminologyManager()
    
    # Add industry-specific terms
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
        description="A computing system inspired by biological neural networks",
        part_of_speech="noun"
    ))
    
    term_manager.add_term(TermDefinition(
        source_term="machine learning",
        translations={
            "es": "aprendizaje automático",
            "fr": "apprentissage automatique",
            "de": "maschinelles Lernen",
            "ja": "機械学習",
            "zh": "机器学习"
        },
        domain="ai",
        description="A field of AI that uses statistical techniques",
        part_of_speech="noun"
    ))
    
    term_manager.add_term(TermDefinition(
        source_term="Avatar Creation Platform",
        translations={},  # Empty translations means keep original
        domain="product",
        description="Product name that should not be translated",
        part_of_speech="noun",
        do_not_translate=True
    ))
    
    # Sample text with terminology to preserve
    text = (
        "Our Avatar Creation Platform uses advanced neural networks and machine learning "
        "to generate realistic avatars. The neural network architecture ensures high-quality results."
    )
    
    # Process the text to find terminology
    processed_text, terms = term_manager.process_text(
        text, args.source_lang, args.target_lang
    )
    
    print(f"Original text: {text}\n")
    print("Detected terminology:")
    for term in terms:
        print(f"  - {term['term']} → {term['translation'] or 'PRESERVE ORIGINAL'}")
        print(f"    (Domain: {term['domain']}, Do not translate: {term['do_not_translate']})")
    
    print("\nSimulating translation with terminology preservation...")
    
    # Initialize translator with terminology preservation
    translator = ContextAwareTranslator()
    options = TranslationOptions(preserve_technical_terms=True)
    
    # In a real implementation, the translator would use the terminology manager
    # For this demo, we'll simulate the process
    translated, metadata = translator.translate_text(
        text, args.source_lang, args.target_lang, options
    )
    
    # For simulation, replace terminology in the translated text
    for term in terms:
        term_text = term['term']
        term_translation = term['translation']
        
        if term['do_not_translate']:
            # For terms that should not be translated, use the original
            if term_text.lower() in translated.lower():
                # This is a simplistic replacement and would be more sophisticated in real implementation
                translated = translated.replace(f"[{args.target_lang}] {term_text}", term_text)
        elif term_translation:
            # For terms with translations, use the terminology database translation
            if term_text.lower() in translated.lower():
                translated = translated.replace(f"[{args.target_lang}] {term_text}", term_translation)
    
    print(f"\nTranslated text with preserved terminology:")
    print(translated)


def demo_cultural_adaptation(args):
    """
    Demonstrate cultural adaptation.
    
    Args:
        args: Command line arguments
    """
    print("\n=== Cultural Adaptation Demo ===\n")
    
    # Initialize the cultural adapter
    adapter = CulturalAdapter()
    
    # Add some cultural references
    adapter.add_reference(CulturalReference(
        source_reference="Super Bowl",
        source_culture="en-US",
        alternatives={
            "es": "Final de la NFL",
            "fr": "Finale du championnat de football américain",
            "de": "American-Football-Finale",
            "ja": "スーパーボウル",
            "zh": "超级碗"
        },
        explanation="The annual championship game of the National Football League (NFL)",
        tags=["sports", "event", "american"]
    ))
    
    adapter.add_reference(CulturalReference(
        source_reference="Fourth of July",
        source_culture="en-US",
        alternatives={
            "es": "Día de la Independencia de Estados Unidos",
            "fr": "Fête nationale américaine",
            "de": "Amerikanischer Unabhängigkeitstag",
            "ja": "独立記念日",
            "zh": "美国独立日"
        },
        explanation="American Independence Day",
        tags=["holiday", "cultural", "american"]
    ))
    
    # Initialize idiomatic expression handler
    idiom_handler = IdiomaticExpressionHandler()
    
    # Add some idiomatic expressions
    idiom_handler.add_expression(IdiomaticExpression(
        source_expression="hit the nail on the head",
        source_language="en",
        translations={
            "es": "dar en el clavo",
            "fr": "taper dans le mille",
            "de": "den Nagel auf den Kopf treffen",
            "ja": "的を射る",
            "zh": "一针见血"
        },
        literal_meaning="To strike a nail precisely on its head with a hammer",
        figurative_meaning="To describe exactly what is causing a situation or problem"
    ))
    
    idiom_handler.add_expression(IdiomaticExpression(
        source_expression="under the weather",
        source_language="en",
        translations={
            "es": "sentirse mal",
            "fr": "être patraque",
            "de": "sich unwohl fühlen",
            "ja": "気分が優れない",
            "zh": "感到不舒服"
        },
        literal_meaning="To be beneath bad weather",
        figurative_meaning="To feel ill or sick"
    ))
    
    # Sample text with cultural references and idioms
    text = (
        "We're planning a Fourth of July party and will watch the Super Bowl highlights. "
        "I think John hit the nail on the head when he said we need more decorations. "
        "Sarah might not come because she's feeling under the weather."
    )
    
    print(f"Original text: {text}\n")
    
    # Process cultural references
    processed_text, adaptations = adapter.process_text(
        text, "en-US", args.target_lang.split('-')[0]  # Use base language without dialect
    )
    
    print("Cultural references detected:")
    for adaptation in adaptations:
        print(f"  - {adaptation['reference']} → {adaptation['alternative'] or 'NO ALTERNATIVE'}")
        print(f"    ({', '.join(adaptation['tags'])})")
    
    # Process idiomatic expressions
    processed_idiom_text, translations = idiom_handler.process_text(
        text, "en", args.target_lang.split('-')[0]  # Use base language without dialect
    )
    
    print("\nIdiomatic expressions detected:")
    for translation in translations:
        print(f"  - {translation['expression']} → {translation['translation'] or 'NO TRANSLATION'}")
        print(f"    (Figurative meaning: {translation['figurative_meaning']})")
    
    print("\nSimulating culturally adapted translation...")
    
    # In a real implementation, both cultural and idiomatic adaptations would be
    # applied during translation. For this demo, we'll simulate the result.
    
    translator = ContextAwareTranslator()
    options = TranslationOptions(
        adapt_cultural_references=True,
        handle_idiomatic_expressions=True
    )
    
    # Simple translation first
    translated, metadata = translator.translate_text(
        text, args.source_lang, args.target_lang, options
    )
    
    # For simulation, replace cultural references in the translated text
    for adaptation in adaptations:
        if adaptation['alternative']:
            translated = translated.replace(
                f"[{args.target_lang}] {adaptation['reference']}", 
                adaptation['alternative']
            )
    
    # For simulation, replace idiomatic expressions in the translated text
    for translation in translations:
        if translation['translation']:
            translated = translated.replace(
                f"[{args.target_lang}] {translation['expression']}", 
                translation['translation']
            )
    
    print(f"\nCulturally adapted translation:")
    print(translated)


def demo_named_entity_preservation(args):
    """
    Demonstrate named entity recognition and preservation.
    
    Args:
        args: Command line arguments
    """
    print("\n=== Named Entity Recognition and Preservation Demo ===\n")
    
    # Initialize the named entity recognizer
    ner = NamedEntityRecognizer()
    
    # Add some custom entities
    ner.add_custom_entity("John Smith", "PERSON")
    ner.add_custom_entity("Acme Corporation", "ORG")
    ner.add_custom_entity("San Francisco", "GPE")
    ner.add_custom_entity("Golden Gate Bridge", "LOC")
    ner.add_custom_entity("iPhone 13", "PRODUCT")
    
    # Sample text with named entities
    text = (
        "John Smith, CEO of Acme Corporation, announced a new office in San Francisco. "
        "The office will have a view of the Golden Gate Bridge. "
        "Employees will receive an iPhone 13 as a welcome gift."
    )
    
    print(f"Original text: {text}\n")
    
    # Process the text to find entities
    processed_text, entities = ner.process_text(text, args.source_lang)
    
    print("Named entities detected:")
    for entity in entities:
        print(f"  - {entity['text']} ({entity['type']})")
    
    print("\nSimulating translation with entity preservation...")
    
    # Initialize translator with entity preservation
    translator = ContextAwareTranslator()
    options = TranslationOptions(preserve_named_entities=True)
    
    # Simple translation first
    translated, metadata = translator.translate_text(
        text, args.source_lang, args.target_lang, options
    )
    
    # For simulation, replace entities in the translated text to preserve them
    for entity in entities:
        entity_text = entity['text']
        # Keep entity in original language
        if entity_text in translated:
            translated = translated.replace(f"[{args.target_lang}] {entity_text}", entity_text)
    
    print(f"\nTranslated text with preserved entities:")
    print(translated)


def demo_timing_preservation(args):
    """
    Demonstrate script timing preservation.
    
    Args:
        args: Command line arguments
    """
    print("\n=== Script Timing Preservation Demo ===\n")
    
    # Initialize the script timing preserver
    timing_preserver = ScriptTimingPreserver()
    
    # Create sample script segments
    segments = [
        TimedSegment(
            text="Welcome to our product demonstration.",
            start_time=0.0,
            end_time=2.0,
            segment_id="1"
        ),
        TimedSegment(
            text="Today we'll show you the amazing features of our avatar creation system.",
            start_time=2.5,
            end_time=5.5,
            segment_id="2"
        ),
        TimedSegment(
            text="Let's start with the face modeling capabilities.",
            start_time=6.0,
            end_time=8.0,
            segment_id="3"
        )
    ]
    
    # Sample translations (these would normally come from a translator)
    translations = {
        "1": "Bienvenido a nuestra demostración de producto.", # Spanish typically longer
        "2": "Hoy le mostraremos las increíbles características de nuestro sistema de creación de avatares.",
        "3": "Comencemos con las capacidades de modelado facial."
    }
    
    print("Original script segments:")
    for segment in segments:
        print(f"  {segment.start_time:.1f}s - {segment.end_time:.1f}s: {segment.text}")
        print(f"    Duration: {segment.duration:.1f}s, Characters/second: {segment.characters_per_second:.1f}")
    
    print("\nTranslated segments (before timing adjustment):")
    for i, segment in enumerate(segments):
        translation = translations[segment.segment_id]
        chars_per_second = len(translation) / segment.duration
        print(f"  {segment.start_time:.1f}s - {segment.end_time:.1f}s: {translation}")
        print(f"    Duration: {segment.duration:.1f}s, Characters/second: {chars_per_second:.1f}")
    
    # Adjust timing for translations
    optimized_segments = timing_preserver.optimize_segments(
        segments, 
        [translations[s.segment_id] for s in segments],
        args.source_lang, 
        args.target_lang
    )
    
    print("\nTranslated segments (after timing adjustment):")
    for segment in optimized_segments:
        print(f"  {segment.start_time:.1f}s - {segment.end_time:.1f}s: {segment.text}")
        print(f"    Duration: {segment.duration:.1f}s, Characters/second: {segment.characters_per_second:.1f}")
    
    # Check for timing issues
    print("\nChecking for timing issues:")
    for segment in optimized_segments:
        issues = timing_preserver.check_timing_issues(segment, args.target_lang)
        if issues["has_issues"]:
            print(f"  Issues detected in segment: {segment.text}")
            if issues["too_fast"]:
                print(f"    Text is too fast: {issues['chars_per_second']:.1f} chars/sec")
                print(f"    Recommended duration: {issues['recommended_duration']:.1f}s")
        else:
            print(f"  No issues in segment: {segment.text[:30]}...")


def demo_technical_vocabulary(args):
    """
    Demonstrate technical vocabulary handling.
    
    Args:
        args: Command line arguments
    """
    print("\n=== Technical Vocabulary Demo ===\n")
    
    # Initialize the technical vocabulary manager
    tech_vocab = TechnicalVocabulary()
    
    # Add technical terms for multiple domains
    tech_vocab.add_term(
        term="deep learning",
        source_language="en",
        translations={
            "es": "aprendizaje profundo",
            "fr": "apprentissage profond",
            "de": "tiefes Lernen",
            "ja": "深層学習",
            "zh": "深度学习"
        },
        domain="ai"
    )
    
    tech_vocab.add_term(
        term="convolutional neural network",
        source_language="en",
        translations={
            "es": "red neuronal convolucional",
            "fr": "réseau neuronal convolutif",
            "de": "Faltungsneuronales Netzwerk",
            "ja": "畳み込みニューラルネットワーク",
            "zh": "卷积神经网络"
        },
        domain="ai"
    )
    
    tech_vocab.add_term(
        term="facial landmark detection",
        source_language="en",
        translations={
            "es": "detección de puntos de referencia faciales",
            "fr": "détection des points de repère du visage",
            "de": "Gesichtslandmarken-Erkennung",
            "ja": "顔のランドマーク検出",
            "zh": "面部特征点检测"
        },
        domain="computer_vision"
    )
    
    # Sample text with technical vocabulary
    text = (
        "Our system uses deep learning and convolutional neural networks for facial landmark detection. "
        "This provides precise control over avatar expressions and animations."
    )
    
    print(f"Original text: {text}\n")
    
    # Look up translations for technical terms
    print("Technical vocabulary translations:")
    tech_terms = ["deep learning", "convolutional neural network", "facial landmark detection"]
    
    for term in tech_terms:
        translation = tech_vocab.get_translation(
            term, args.source_lang, args.target_lang.split('-')[0]
        )
        print(f"  - {term} → {translation or 'NOT FOUND'}")
    
    print("\nSimulating translation with technical vocabulary...")
    
    # Initialize translator
    translator = ContextAwareTranslator()
    options = TranslationOptions(preserve_technical_terms=True)
    
    # Simple translation first
    translated, metadata = translator.translate_text(
        text, args.source_lang, args.target_lang, options
    )
    
    # For simulation, replace technical terms in the translated text
    for term in tech_terms:
        translation = tech_vocab.get_translation(
            term, args.source_lang, args.target_lang.split('-')[0]
        )
        if translation:
            # This is a simplistic replacement and would be more sophisticated in real implementation
            translated = translated.replace(f"[{args.target_lang}] {term}", translation)
    
    print(f"\nTranslated text with technical vocabulary:")
    print(translated)


def main():
    """Main function to run the video translation module demo."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Video Translation Module Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--source-lang", type=str, default="en",
                      help="Source language code")
    parser.add_argument("--target-lang", type=str, default="es",
                      help="Target language code")
    parser.add_argument("--demo", type=str, choices=[
                        "all", "basic", "context", "terminology", 
                        "cultural", "entities", "timing", "technical"
                      ], default="all",
                      help="Which demo to run")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Video Translation Module Demo")
    print("=" * 80)
    print(f"Source language: {args.source_lang}")
    print(f"Target language: {args.target_lang}")
    
    # Create data directories
    create_data_directories()
    
    # Run the selected demo
    if args.demo in ["all", "basic"]:
        demo_basic_translation(args)
    
    if args.demo in ["all", "context"]:
        demo_context_aware_translation(args)
    
    if args.demo in ["all", "terminology"]:
        demo_terminology_preservation(args)
    
    if args.demo in ["all", "cultural"]:
        demo_cultural_adaptation(args)
    
    if args.demo in ["all", "entities"]:
        demo_named_entity_preservation(args)
    
    if args.demo in ["all", "timing"]:
        demo_timing_preservation(args)
    
    if args.demo in ["all", "technical"]:
        demo_technical_vocabulary(args)
    
    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


# For usage from TermDefinition class in the terminology module
from app.avatar_creation.video_translation.terminology import TermDefinition

if __name__ == "__main__":
    main() 