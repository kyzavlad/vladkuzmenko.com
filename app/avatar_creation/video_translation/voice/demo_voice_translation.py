#!/usr/bin/env python3
"""
Voice Translation System Demo

This script demonstrates the features of the Voice Translation System,
including voice characteristic preservation, emotion transfer, prosody modeling,
and multi-speaker support.

Usage:
    python demo_voice_translation.py --mode [mode] --input [input_audio] --output [output_path]

Modes:
    basic - Basic voice translation with voice characteristic preservation
    emotion - Voice translation with emotion transfer
    prosody - Voice translation with prosody modeling
    multi - Multi-speaker voice translation
    all - Run all demos in sequence
"""

import os
import sys
import argparse
import time
import numpy as np
import librosa
from typing import Dict, List, Optional, Any

from app.avatar_creation.video_translation.voice.voice_translator import VoiceTranslator
from app.avatar_creation.video_translation.voice.emotion_transfer import EmotionTransferSystem
from app.avatar_creation.video_translation.voice.prosody_modeling import ProsodyModeler
from app.avatar_creation.video_translation.voice.speaker_separation import SpeakerSeparator, SpeakerSegment


class VoiceTranslationDemo:
    """
    Demonstration class for Voice Translation System capabilities.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the voice translation demo.
        
        Args:
            verbose: Whether to print detailed information
        """
        self.verbose = verbose
        print("Initializing Voice Translation Demo")
        
        # Create the voice translator
        self.voice_translator = VoiceTranslator(
            voice_model_path="models/voice/translator_model",
            emotion_model_path="models/emotion/transfer_model",
            prosody_model_path="models/prosody/model_weights"
        )
        
        # Create the speaker separator for multi-speaker demos
        self.speaker_separator = SpeakerSeparator(
            model_path="models/speaker/separator_model"
        )
        
        # Sample transcripts and translations for demo
        self.sample_transcripts = {
            "en": {
                "basic": "This is a demonstration of voice translation.",
                "emotion": "I'm really excited about this new technology! It's amazing!",
                "prosody": "Voice translation should preserve natural rhythm and intonation. It makes the speech sound more natural.",
                "multi": [
                    "Hello, how are you today?",
                    "I'm doing well, thank you for asking.",
                    "That's great to hear. The weather is nice today."
                ]
            }
        }
        
        self.sample_translations = {
            "es": {
                "basic": "Esta es una demostración de traducción de voz.",
                "emotion": "¡Estoy muy emocionado por esta nueva tecnología! ¡Es increíble!",
                "prosody": "La traducción de voz debe preservar el ritmo y la entonación naturales. Hace que el habla suene más natural.",
                "multi": [
                    "Hola, ¿cómo estás hoy?",
                    "Estoy bien, gracias por preguntar.",
                    "Me alegra oír eso. El clima está agradable hoy."
                ]
            },
            "fr": {
                "basic": "Ceci est une démonstration de traduction vocale.",
                "emotion": "Je suis vraiment excité par cette nouvelle technologie! C'est incroyable!",
                "prosody": "La traduction vocale doit préserver le rythme naturel et l'intonation. Cela rend la parole plus naturelle.",
                "multi": [
                    "Bonjour, comment allez-vous aujourd'hui?",
                    "Je vais bien, merci de demander.",
                    "C'est super à entendre. Le temps est agréable aujourd'hui."
                ]
            }
        }
        
        print("Voice Translation Demo initialized")
    
    def run_basic_demo(self, 
                     input_path: str, 
                     output_path: str, 
                     source_lang: str = "en", 
                     target_lang: str = "es") -> str:
        """
        Run the basic voice translation demo with voice characteristic preservation.
        
        Args:
            input_path: Path to the input audio file
            output_path: Path for the output audio file
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Path to the generated audio file
        """
        print(f"\n{'='*40}")
        print(f"Running Basic Voice Translation Demo")
        print(f"{'='*40}")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Languages: {source_lang} → {target_lang}")
        
        # Get sample transcript and translation
        transcript = self.sample_transcripts.get(source_lang, {}).get("basic", "Sample text")
        translated_text = self.sample_translations.get(target_lang, {}).get("basic", "Sample translation")
        
        # Run voice translation
        result_path = self.voice_translator.translate_voice(
            input_path, transcript, translated_text, output_path, source_lang, target_lang
        )
        
        print(f"Basic voice translation completed: {result_path}")
        return result_path
    
    def run_emotion_demo(self, 
                       input_path: str, 
                       output_path: str, 
                       source_lang: str = "en", 
                       target_lang: str = "es") -> str:
        """
        Run the emotion transfer demo.
        
        Args:
            input_path: Path to the input audio file
            output_path: Path for the output audio file
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Path to the generated audio file
        """
        print(f"\n{'='*40}")
        print(f"Running Emotion Transfer Demo")
        print(f"{'='*40}")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Languages: {source_lang} → {target_lang}")
        
        # Get sample transcript and translation
        transcript = self.sample_transcripts.get(source_lang, {}).get("emotion", "Sample emotional text")
        translated_text = self.sample_translations.get(target_lang, {}).get("emotion", "Sample emotional translation")
        
        # Load audio
        try:
            y, sr = librosa.load(input_path, sr=None)
        except Exception as e:
            print(f"Error loading audio file {input_path}: {e}")
            return output_path
        
        # Run voice translation
        result_path = self.voice_translator.translate_voice(
            input_path, transcript, translated_text, output_path, source_lang, target_lang
        )
        
        # Print additional information about emotion transfer
        emotion_scores = self.voice_translator.detect_emotion(input_path)
        
        print("\nDetected emotions:")
        for emotion, score in sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  {emotion}: {score:.2f}")
        
        print(f"Emotion transfer demo completed: {result_path}")
        return result_path
    
    def run_prosody_demo(self, 
                       input_path: str, 
                       output_path: str, 
                       source_lang: str = "en", 
                       target_lang: str = "es") -> str:
        """
        Run the prosody modeling demo.
        
        Args:
            input_path: Path to the input audio file
            output_path: Path for the output audio file
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Path to the generated audio file
        """
        print(f"\n{'='*40}")
        print(f"Running Prosody Modeling Demo")
        print(f"{'='*40}")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Languages: {source_lang} → {target_lang}")
        
        # Get sample transcript and translation
        transcript = self.sample_transcripts.get(source_lang, {}).get("prosody", "Sample text with prosody")
        translated_text = self.sample_translations.get(target_lang, {}).get("prosody", "Sample translation with prosody")
        
        # Load audio
        try:
            y, sr = librosa.load(input_path, sr=None)
        except Exception as e:
            print(f"Error loading audio file {input_path}: {e}")
            return output_path
        
        # Analyze prosody
        prosody_info = self.voice_translator.analyze_prosody(input_path, transcript, source_lang)
        
        if self.verbose:
            print("\nProsody analysis:")
            print(f"  Speech rate: {prosody_info.get('speech_rate', 0):.2f}")
            print(f"  Pauses: {len(prosody_info.get('pauses', []))}")
            print(f"  Speech units: {len(prosody_info.get('speech_units', []))}")
        
        # Run voice translation
        result_path = self.voice_translator.translate_voice(
            input_path, transcript, translated_text, output_path, source_lang, target_lang
        )
        
        print(f"Prosody modeling demo completed: {result_path}")
        return result_path
    
    def run_multi_speaker_demo(self, 
                            input_path: str, 
                            output_path: str, 
                            source_lang: str = "en", 
                            target_lang: str = "es") -> str:
        """
        Run the multi-speaker voice translation demo.
        
        Args:
            input_path: Path to the input audio file
            output_path: Path for the output audio file
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Path to the generated audio file
        """
        print(f"\n{'='*40}")
        print(f"Running Multi-Speaker Voice Translation Demo")
        print(f"{'='*40}")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Languages: {source_lang} → {target_lang}")
        
        # Get sample transcripts and translations
        transcripts = self.sample_transcripts.get(source_lang, {}).get("multi", ["Speaker 1", "Speaker 2", "Speaker 3"])
        translations = self.sample_translations.get(target_lang, {}).get("multi", ["Speaker 1", "Speaker 2", "Speaker 3"])
        
        # Ensure we have the same number of texts
        num_speakers = min(len(transcripts), len(translations))
        transcripts = transcripts[:num_speakers]
        translations = translations[:num_speakers]
        
        # Process the multi-speaker audio
        segments, profiles = self.speaker_separator.process_audio(input_path, num_speakers=num_speakers)
        
        if not segments:
            print("Error: No speaker segments detected")
            return output_path
        
        # Create segment data with transcripts
        speaker_segments = []
        
        # Assign transcripts to segments based on speaker
        speaker_texts = {}
        for i, (transcript, translation) in enumerate(zip(transcripts, translations)):
            if i < num_speakers:
                speaker_texts[i] = {"transcript": transcript, "translation": translation}
        
        # Create segment data
        for segment in segments:
            speaker_id = segment.speaker_id
            if speaker_id in speaker_texts:
                segment_data = {
                    "speaker_id": speaker_id,
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "text": speaker_texts[speaker_id]["transcript"]
                }
                speaker_segments.append(segment_data)
        
        # Create full transcript and translation
        full_transcript = " ".join(speaker_texts[i]["transcript"] for i in range(num_speakers) if i in speaker_texts)
        full_translation = " ".join(speaker_texts[i]["translation"] for i in range(num_speakers) if i in speaker_texts)
        
        # Translate the multi-speaker audio
        result_path = self.voice_translator.process_multi_speaker(
            input_path, speaker_segments, full_transcript, full_translation,
            output_path, source_lang, target_lang
        )
        
        if self.verbose:
            print("\nSpeaker profiles:")
            for speaker_id, profile in profiles.items():
                if speaker_id in speaker_texts:
                    print(f"  Speaker {speaker_id}:")
                    print(f"    Gender: {profile.gender}")
                    print(f"    Duration: {profile.duration:.2f}s")
                    print(f"    Text: {speaker_texts[speaker_id]['transcript']}")
                    print(f"    Translation: {speaker_texts[speaker_id]['translation']}")
        
        print(f"Multi-speaker voice translation completed: {result_path}")
        return result_path
    
    def run_all_demos(self, 
                    input_path: str, 
                    output_dir: str, 
                    source_lang: str = "en", 
                    target_lang: str = "es") -> List[str]:
        """
        Run all voice translation demos.
        
        Args:
            input_path: Path to the input audio file
            output_dir: Directory for the output files
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            List of paths to the generated audio files
        """
        print(f"\n{'='*40}")
        print(f"Running All Voice Translation Demos")
        print(f"{'='*40}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output paths
        output_paths = {
            "basic": os.path.join(output_dir, "basic_voice_translation.wav"),
            "emotion": os.path.join(output_dir, "emotion_transfer.wav"),
            "prosody": os.path.join(output_dir, "prosody_modeling.wav"),
            "multi": os.path.join(output_dir, "multi_speaker.wav")
        }
        
        # Run each demo
        results = []
        results.append(self.run_basic_demo(input_path, output_paths["basic"], source_lang, target_lang))
        results.append(self.run_emotion_demo(input_path, output_paths["emotion"], source_lang, target_lang))
        results.append(self.run_prosody_demo(input_path, output_paths["prosody"], source_lang, target_lang))
        
        # For multi-speaker, use the same input or a different one if specified
        multi_input = input_path
        results.append(self.run_multi_speaker_demo(multi_input, output_paths["multi"], source_lang, target_lang))
        
        print(f"\n{'='*40}")
        print(f"All demos completed successfully!")
        print(f"{'='*40}")
        print("Generated audio files:")
        for i, path in enumerate(results):
            print(f"  {i+1}. {path}")
        
        return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Voice Translation System Demo")
    
    parser.add_argument("--mode", type=str, default="all",
                      choices=["basic", "emotion", "prosody", "multi", "all"],
                      help="Demo mode to run")
    
    parser.add_argument("--input", type=str, default="example/audio/sample.wav",
                      help="Path to the input audio file")
    
    parser.add_argument("--output", type=str, default="example/output",
                      help="Path for the output audio file or directory (for 'all' mode)")
    
    parser.add_argument("--source-lang", type=str, default="en",
                      help="Source language code")
    
    parser.add_argument("--target-lang", type=str, default="es",
                      help="Target language code")
    
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose output")
    
    return parser.parse_args()


def main():
    """Main function to run the demo."""
    args = parse_args()
    
    # Create the demo
    demo = VoiceTranslationDemo(verbose=args.verbose)
    
    # Run the selected demo mode
    if args.mode == "basic":
        output_path = args.output
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, "basic_voice_translation.wav")
        
        demo.run_basic_demo(args.input, output_path, args.source_lang, args.target_lang)
    
    elif args.mode == "emotion":
        output_path = args.output
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, "emotion_transfer.wav")
        
        demo.run_emotion_demo(args.input, output_path, args.source_lang, args.target_lang)
    
    elif args.mode == "prosody":
        output_path = args.output
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, "prosody_modeling.wav")
        
        demo.run_prosody_demo(args.input, output_path, args.source_lang, args.target_lang)
    
    elif args.mode == "multi":
        output_path = args.output
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, "multi_speaker.wav")
        
        demo.run_multi_speaker_demo(args.input, output_path, args.source_lang, args.target_lang)
    
    elif args.mode == "all":
        demo.run_all_demos(args.input, args.output, args.source_lang, args.target_lang)
    
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main() 