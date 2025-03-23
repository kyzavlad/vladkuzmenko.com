#!/usr/bin/env python
"""
Voice Cloning System Demo

This script demonstrates the voice cloning functionality, including:
- Voice characteristic extraction
- Speaker embedding
- Neural vocoder
- Voice cloning and synthesis
- Prosody and style transfer
- Voice consistency verification

Usage examples:
    python -m app.avatar_creation.voice_cloning.demo --mode clone --audio samples/voice.wav --name "user_voice"
    python -m app.avatar_creation.voice_cloning.demo --mode synthesize --voice "user_voice" --text "Hello, this is my cloned voice."
    python -m app.avatar_creation.voice_cloning.demo --mode transfer_prosody --source samples/emotion.wav --target "user_voice"
    python -m app.avatar_creation.voice_cloning.demo --mode verify --original samples/voice.wav --synthesized output/synthesized.wav
"""

import os
import argparse
import time
from app.avatar_creation.voice_cloning import (
    VoiceCloner, VoiceCharacteristicExtractor,
    SpeakerEmbedding, NeuralVocoder
)

def clone_voice(args):
    """
    Clone a voice from an audio file.
    
    Args:
        args: Command line arguments
    """
    print(f"Cloning voice from {args.audio}")
    
    # Initialize voice cloner
    cloner = VoiceCloner(
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        use_gpu=not args.cpu
    )
    
    # Clone the voice
    result = cloner.clone_voice(
        audio_path=args.audio,
        voice_name=args.name
    )
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
    
    print(f"Voice cloned successfully in {result['processing_time']:.2f} seconds")
    print(f"Voice data saved to {result['voice_dir']}")
    
    # Print voice characteristics
    if 'characteristics' in result and 'pitch' in result['characteristics']:
        pitch = result['characteristics']['pitch']
        print(f"Voice pitch statistics:")
        print(f"  - Mean: {pitch['mean']:.1f} Hz")
        print(f"  - Range: {pitch['range']:.1f} Hz")
        print(f"  - Variation: {pitch['variation']:.3f}")
    
    if 'characteristics' in result and 'quality' in result['characteristics']:
        quality = result['characteristics']['quality']
        print(f"Voice quality metrics:")
        print(f"  - Harmonics-to-noise ratio: {quality['hnr']:.1f} dB")
        print(f"  - Jitter: {quality['jitter']:.3f}")
        print(f"  - Shimmer: {quality['shimmer']:.3f}")

def synthesize_speech(args):
    """
    Synthesize speech using a cloned voice.
    
    Args:
        args: Command line arguments
    """
    print(f"Synthesizing speech using voice '{args.voice}'")
    print(f"Text: {args.text}")
    
    # Initialize voice cloner
    cloner = VoiceCloner(
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        use_gpu=not args.cpu
    )
    
    # First check if we need to load the voice
    if args.voice not in cloner.get_cloned_voices():
        voice_dir = os.path.join(args.output_dir, args.voice)
        if os.path.exists(voice_dir):
            # Load the voice
            cloner.load_voice(voice_dir)
        else:
            print(f"Error: Voice '{args.voice}' not found. Clone it first.")
            return
    
    # Synthesize speech
    result = cloner.synthesize_speech(
        text=args.text,
        voice_name=args.voice,
        output_path=args.output,
        emotion=args.emotion,
        speaking_rate=args.rate,
        pitch_shift=args.pitch_shift
    )
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
    
    print(f"Speech synthesized successfully in {result['processing_time']:.2f} seconds")
    print(f"Audio saved to {result['output_path']}")
    print(f"Audio length: {result['audio_length']:.2f} seconds")
    print(f"Real-time factor: {result['real_time_factor']:.2f}x")

def transfer_prosody(args):
    """
    Transfer prosody from a source audio to a target voice.
    
    Args:
        args: Command line arguments
    """
    print(f"Transferring prosody from {args.source} to voice '{args.target}'")
    
    # Initialize voice cloner
    cloner = VoiceCloner(
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        use_gpu=not args.cpu
    )
    
    # First check if we need to load the voice
    if args.target not in cloner.get_cloned_voices():
        voice_dir = os.path.join(args.output_dir, args.target)
        if os.path.exists(voice_dir):
            # Load the voice
            cloner.load_voice(voice_dir)
        else:
            print(f"Error: Voice '{args.target}' not found. Clone it first.")
            return
    
    # Transfer prosody
    result = cloner.transfer_prosody(
        source_audio_path=args.source,
        target_voice_name=args.target,
        output_path=args.output
    )
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
    
    print(f"Prosody transfer completed successfully in {result['processing_time']:.2f} seconds")
    print(f"Audio saved to {result['output_path']}")
    print(f"Audio length: {result['audio_length']:.2f} seconds")
    print(f"Real-time factor: {result['real_time_factor']:.2f}x")

def transfer_style(args):
    """
    Transfer speaking style from a source audio to a target voice.
    
    Args:
        args: Command line arguments
    """
    print(f"Transferring style from {args.source} to voice '{args.target}'")
    print(f"Text: {args.text}")
    print(f"Style strength: {args.strength}")
    
    # Initialize voice cloner
    cloner = VoiceCloner(
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        use_gpu=not args.cpu
    )
    
    # First check if we need to load the voice
    if args.target not in cloner.get_cloned_voices():
        voice_dir = os.path.join(args.output_dir, args.target)
        if os.path.exists(voice_dir):
            # Load the voice
            cloner.load_voice(voice_dir)
        else:
            print(f"Error: Voice '{args.target}' not found. Clone it first.")
            return
    
    # Transfer style
    result = cloner.transfer_style(
        source_style_path=args.source,
        target_voice_name=args.target,
        text=args.text,
        style_strength=args.strength,
        output_path=args.output
    )
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
    
    print(f"Style transfer completed successfully in {result['processing_time']:.2f} seconds")
    print(f"Audio saved to {result['output_path']}")
    print(f"Audio length: {result['audio_length']:.2f} seconds")
    print(f"Real-time factor: {result['real_time_factor']:.2f}x")

def verify_voice_consistency(args):
    """
    Verify consistency between original and synthesized voice.
    
    Args:
        args: Command line arguments
    """
    print(f"Verifying voice consistency")
    print(f"Original audio: {args.original}")
    print(f"Synthesized audio: {args.synthesized}")
    
    # Initialize voice cloner
    cloner = VoiceCloner(
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        use_gpu=not args.cpu
    )
    
    # Verify voice consistency
    metrics = cloner.verify_voice_consistency(
        original_audio_path=args.original,
        synthesized_audio_path=args.synthesized
    )
    
    if 'error' in metrics:
        print(f"Error: {metrics['error']}")
        return
    
    print("\nVoice Consistency Metrics:")
    print(f"  - Feature cosine similarity: {metrics['feature_cosine_similarity']:.4f}")
    print(f"  - Embedding cosine similarity: {metrics['embedding_cosine_similarity']:.4f}")
    print(f"  - Pitch similarity: {metrics['pitch_similarity']:.4f}")
    print(f"  - Rhythm similarity: {metrics['rhythm_similarity']:.4f}")
    print(f"  - Voice quality similarity: {metrics['voice_quality_similarity']:.4f}")
    print(f"  - Overall similarity: {metrics['overall_similarity']:.4f}")
    
    # Interpret the results
    if metrics['overall_similarity'] > 0.8:
        print("\nInterpretation: Excellent voice similarity! The synthesized voice is very close to the original.")
    elif metrics['overall_similarity'] > 0.6:
        print("\nInterpretation: Good voice similarity. The synthesized voice is recognizable as the same person.")
    elif metrics['overall_similarity'] > 0.4:
        print("\nInterpretation: Moderate voice similarity. Some characteristics are preserved but others differ.")
    else:
        print("\nInterpretation: Low voice similarity. The synthesized voice differs significantly from the original.")

def extract_characteristics(args):
    """
    Extract voice characteristics from an audio file.
    
    Args:
        args: Command line arguments
    """
    print(f"Extracting voice characteristics from {args.audio}")
    
    # Initialize extractor
    extractor = VoiceCharacteristicExtractor(
        sample_rate=args.sample_rate,
        min_sample_duration=args.min_duration,
        use_gpu=not args.cpu
    )
    
    # Extract characteristics
    result = extractor.extract_characteristics(
        audio_path=args.audio,
        save_features=True,
        output_dir=args.output_dir
    )
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
    
    print(f"Voice characteristics extracted in {result['processing_time']:.2f} seconds")
    
    # Print pitch statistics
    if 'pitch' in result:
        pitch = result['pitch']
        print("\nPitch Statistics:")
        print(f"  - Mean: {pitch['mean']:.1f} Hz")
        print(f"  - Median: {pitch['median']:.1f} Hz")
        print(f"  - Standard deviation: {pitch['std']:.1f} Hz")
        print(f"  - Range: {pitch['range']:.1f} Hz")
        print(f"  - Voiced ratio: {pitch['voiced_ratio']:.3f}")
        print(f"  - Variation: {pitch['variation']:.3f}")
    
    # Print voice quality
    if 'quality' in result:
        quality = result['quality']
        print("\nVoice Quality Metrics:")
        print(f"  - Harmonics-to-noise ratio: {quality['hnr']:.1f} dB")
        print(f"  - Jitter: {quality['jitter']:.3f}")
        print(f"  - Shimmer: {quality['shimmer']:.3f}")
    
    # Print temporal features
    if 'temporal' in result:
        temporal = result['temporal']
        print("\nTemporal Features:")
        print(f"  - Speaking rate (approx): {temporal.get('syllable_rate', 0):.1f} syllables/sec")
        print(f"  - Zero crossing rate: {temporal.get('zcr_mean', 0):.3f}")
    
    # Print feature vector info
    if 'feature_vector' in result:
        feature_vector = result['feature_vector']
        print(f"\nFeature vector dimensions: {feature_vector.shape}")
        print(f"Feature path: {result.get('feature_path', 'Not saved')}")

def extract_embedding(args):
    """
    Extract speaker embedding from an audio file.
    
    Args:
        args: Command line arguments
    """
    print(f"Extracting speaker embedding from {args.audio}")
    
    # Initialize embedding extractor
    embedding_extractor = SpeakerEmbedding(
        model_path=args.model_path,
        use_gpu=not args.cpu,
        sample_rate=args.sample_rate
    )
    
    # Extract embedding
    result = embedding_extractor.extract_embedding(
        audio_path=args.audio,
        save_embedding=True,
        output_dir=args.output_dir
    )
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
    
    print(f"Speaker embedding extracted in {result['processing_time']:.2f} seconds")
    print(f"Embedding dimensions: {result['embedding_dim']}")
    print(f"Embedding path: {result.get('embedding_path', 'Not saved')}")
    
    # If we have two embeddings to compare
    if args.compare_with:
        print(f"\nComparing with embedding from {args.compare_with}")
        
        # Extract embedding for comparison
        compare_result = embedding_extractor.extract_embedding(
            audio_path=args.compare_with,
            save_embedding=False
        )
        
        if 'error' in compare_result:
            print(f"Error extracting comparison embedding: {compare_result['error']}")
            return
        
        # Compare embeddings
        similarity = embedding_extractor.compare_embeddings(
            result['embedding'],
            compare_result['embedding']
        )
        
        print("\nSimilarity Metrics:")
        print(f"  - Cosine similarity: {similarity['cosine_similarity']:.4f}")
        print(f"  - Euclidean distance: {similarity['euclidean_distance']:.4f}")
        print(f"  - PLDA score: {similarity['plda_score']:.4f}")
        
        # Interpret the results
        if similarity['cosine_similarity'] > 0.85:
            print("\nInterpretation: Extremely high similarity. Almost certainly the same speaker.")
        elif similarity['cosine_similarity'] > 0.7:
            print("\nInterpretation: High similarity. Very likely the same speaker.")
        elif similarity['cosine_similarity'] > 0.5:
            print("\nInterpretation: Moderate similarity. Could be the same speaker or similar voices.")
        else:
            print("\nInterpretation: Low similarity. Probably different speakers.")

def synthesize_vocoder(args):
    """
    Synthesize audio using the neural vocoder directly.
    
    Args:
        args: Command line arguments
    """
    print(f"Synthesizing audio using the neural vocoder")
    
    # This is a demonstration - in a real scenario, we would need a mel spectrogram
    # Here we're using random noise as a placeholder
    import numpy as np
    
    # Create a random mel spectrogram for demonstration
    # In a real use case, this would come from a text-to-mel model
    mel_length = int(args.duration * 1000 / 12.5)  # Assuming 12.5ms per frame
    mel_spec = np.random.random((80, mel_length)) * 2 - 1
    
    print(f"Generated placeholder mel spectrogram with shape {mel_spec.shape}")
    
    # Initialize vocoder
    vocoder = NeuralVocoder(
        model_path=args.model_path,
        use_gpu=not args.cpu,
        sample_rate=args.sample_rate
    )
    
    # Synthesize audio
    waveform, metadata = vocoder.synthesize(
        mel_spectrogram=mel_spec,
        save_path=args.output,
        target_text="Vocoder demonstration"
    )
    
    print(f"Audio synthesized in {metadata['generation_time']:.2f} seconds")
    print(f"Audio length: {metadata['audio_length']:.2f} seconds")
    print(f"Real-time factor: {metadata['real_time_factor']:.2f}x")
    print(f"Audio saved to {args.output}")
    
    # Get performance stats
    stats = vocoder.get_performance_stats()
    
    if stats:
        print("\nVocoder Performance Statistics:")
        print(f"  - Mean RTF: {stats.get('mean_rtf', 0):.2f}x")
        print(f"  - Mean generation time: {stats.get('mean_generation_time', 0):.2f} seconds")

def main():
    """Main function to parse arguments and call appropriate functions."""
    parser = argparse.ArgumentParser(
        description="Voice Cloning System Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Global arguments
    parser.add_argument('--models_dir', type=str, default=None,
                        help='Directory containing pre-trained models')
    parser.add_argument('--output_dir', type=str, default='output/voice_cloning',
                        help='Directory for output files')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage (no GPU)')
    parser.add_argument('--sample_rate', type=int, default=22050,
                        help='Audio sample rate')
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Clone voice mode
    clone_parser = subparsers.add_parser('clone', help='Clone a voice from an audio file')
    clone_parser.add_argument('--audio', type=str, required=True,
                             help='Path to the audio file')
    clone_parser.add_argument('--name', type=str, required=True,
                             help='Name for the cloned voice')
    
    # Synthesize speech mode
    synth_parser = subparsers.add_parser('synthesize', help='Synthesize speech using a cloned voice')
    synth_parser.add_argument('--voice', type=str, required=True,
                             help='Name of the cloned voice to use')
    synth_parser.add_argument('--text', type=str, required=True,
                             help='Text to synthesize')
    synth_parser.add_argument('--output', type=str, default=None,
                             help='Path to save the synthesized audio')
    synth_parser.add_argument('--emotion', type=str, default=None,
                             help='Emotion to apply (happy, sad, angry, etc.)')
    synth_parser.add_argument('--rate', type=float, default=1.0,
                             help='Speaking rate multiplier')
    synth_parser.add_argument('--pitch_shift', type=float, default=0.0,
                             help='Pitch shift in semitones')
    
    # Transfer prosody mode
    prosody_parser = subparsers.add_parser('transfer_prosody', help='Transfer prosody to a voice')
    prosody_parser.add_argument('--source', type=str, required=True,
                               help='Path to the source audio with desired prosody')
    prosody_parser.add_argument('--target', type=str, required=True,
                               help='Name of the target cloned voice')
    prosody_parser.add_argument('--output', type=str, default=None,
                               help='Path to save the output audio')
    
    # Transfer style mode
    style_parser = subparsers.add_parser('transfer_style', help='Transfer speaking style to a voice')
    style_parser.add_argument('--source', type=str, required=True,
                             help='Path to the audio with desired speaking style')
    style_parser.add_argument('--target', type=str, required=True,
                             help='Name of the target cloned voice')
    style_parser.add_argument('--text', type=str, required=True,
                             help='Text to synthesize')
    style_parser.add_argument('--strength', type=float, default=0.5,
                             help='Strength of style transfer (0.0-1.0)')
    style_parser.add_argument('--output', type=str, default=None,
                             help='Path to save the output audio')
    
    # Verify voice consistency mode
    verify_parser = subparsers.add_parser('verify', help='Verify voice consistency')
    verify_parser.add_argument('--original', type=str, required=True,
                              help='Path to the original voice audio')
    verify_parser.add_argument('--synthesized', type=str, required=True,
                              help='Path to the synthesized voice audio')
    
    # Extract voice characteristics mode
    chars_parser = subparsers.add_parser('characteristics', help='Extract voice characteristics')
    chars_parser.add_argument('--audio', type=str, required=True,
                             help='Path to the audio file')
    chars_parser.add_argument('--min_duration', type=float, default=15.0,
                             help='Minimum sample duration in seconds')
    
    # Extract speaker embedding mode
    embed_parser = subparsers.add_parser('embedding', help='Extract speaker embedding')
    embed_parser.add_argument('--audio', type=str, required=True,
                             help='Path to the audio file')
    embed_parser.add_argument('--model_path', type=str, default=None,
                             help='Path to the x-vector model')
    embed_parser.add_argument('--compare_with', type=str, default=None,
                             help='Path to audio file to compare embedding with')
    
    # Synthesize with vocoder mode
    vocoder_parser = subparsers.add_parser('vocoder', help='Synthesize using neural vocoder')
    vocoder_parser.add_argument('--model_path', type=str, default=None,
                               help='Path to the WaveRNN model')
    vocoder_parser.add_argument('--output', type=str, required=True,
                               help='Path to save the output audio')
    vocoder_parser.add_argument('--duration', type=float, default=3.0,
                               help='Duration of the audio to generate in seconds')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call appropriate function based on mode
    if args.mode == 'clone':
        clone_voice(args)
    elif args.mode == 'synthesize':
        synthesize_speech(args)
    elif args.mode == 'transfer_prosody':
        transfer_prosody(args)
    elif args.mode == 'transfer_style':
        transfer_style(args)
    elif args.mode == 'verify':
        verify_voice_consistency(args)
    elif args.mode == 'characteristics':
        extract_characteristics(args)
    elif args.mode == 'embedding':
        extract_embedding(args)
    elif args.mode == 'vocoder':
        synthesize_vocoder(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 