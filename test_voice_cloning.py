#!/usr/bin/env python
"""
Voice Cloning System Test Script

This script demonstrates the voice cloning system functionality by:
1. Cloning a voice from a sample audio file
2. Synthesizing speech with the cloned voice
3. Performing prosody transfer
4. Verifying voice consistency

Note: This is a demonstration with placeholder audio processing.
      In a real-world scenario, you would use actual voice samples.
"""

import os
from app.avatar_creation.voice_cloning import VoiceCloner

def setup_test_files():
    """Creates placeholder audio files for testing if they don't exist"""
    os.makedirs('samples/voice_samples', exist_ok=True)
    
    # Create dummy test files with ffmpeg if they don't exist
    # Generate 5 seconds of silence
    if not os.path.exists('samples/voice_samples/sample_voice.wav'):
        os.system("ffmpeg -f lavfi -i anullsrc=r=44100:cl=mono -t 15 -q:a 9 -acodec pcm_s16le samples/voice_samples/sample_voice.wav -y")
    
    if not os.path.exists('samples/voice_samples/source_prosody.wav'):
        os.system("ffmpeg -f lavfi -i anullsrc=r=44100:cl=mono -t 10 -q:a 9 -acodec pcm_s16le samples/voice_samples/source_prosody.wav -y")
    
    if not os.path.exists('samples/voice_samples/source_style.wav'):
        os.system("ffmpeg -f lavfi -i anullsrc=r=44100:cl=mono -t 8 -q:a 9 -acodec pcm_s16le samples/voice_samples/source_style.wav -y")

def test_voice_cloning():
    """Main function to test voice cloning functionality"""
    print("\n=== Voice Cloning System Test ===\n")
    
    # Setup test files
    setup_test_files()
    
    # Initialize voice cloner
    cloner = VoiceCloner(
        models_dir=None,  # Using placeholder implementation
        output_dir='output/voice_cloning',
        use_gpu=False  # Set to True to use GPU if available
    )
    
    # Step 1: Clone a voice
    print("\n--- Step 1: Cloning Voice ---\n")
    voice_name = "test_voice"
    voice_result = cloner.clone_voice(
        audio_path='samples/voice_samples/sample_voice.wav',
        voice_name=voice_name
    )
    
    if 'error' in voice_result:
        print(f"Error cloning voice: {voice_result['error']}")
        return
    
    # Step 2: Synthesize speech
    print("\n--- Step 2: Synthesizing Speech ---\n")
    synth_result = cloner.synthesize_speech(
        text="Hello world! This is a test of the voice cloning system.",
        voice_name=voice_name,
        output_path='output/voice_cloning/synthesized_speech.wav'
    )
    
    if 'error' in synth_result:
        print(f"Error synthesizing speech: {synth_result['error']}")
        return
    
    # Step 3: Transfer prosody
    print("\n--- Step 3: Transferring Prosody ---\n")
    prosody_result = cloner.transfer_prosody(
        source_audio_path='samples/voice_samples/source_prosody.wav',
        target_voice_name=voice_name,
        output_path='output/voice_cloning/prosody_transfer.wav'
    )
    
    if 'error' in prosody_result:
        print(f"Error transferring prosody: {prosody_result['error']}")
        return
    
    # Step 4: Transfer style
    print("\n--- Step 4: Transferring Style ---\n")
    style_result = cloner.transfer_style(
        source_style_path='samples/voice_samples/source_style.wav',
        target_voice_name=voice_name,
        text="This is text with a transferred speaking style.",
        style_strength=0.7,
        output_path='output/voice_cloning/style_transfer.wav'
    )
    
    if 'error' in style_result:
        print(f"Error transferring style: {style_result['error']}")
        return
    
    # Step 5: Verify voice consistency
    print("\n--- Step 5: Verifying Voice Consistency ---\n")
    consistency_metrics = cloner.verify_voice_consistency(
        original_audio_path='samples/voice_samples/sample_voice.wav',
        synthesized_audio_path='output/voice_cloning/synthesized_speech.wav'
    )
    
    if 'error' in consistency_metrics:
        print(f"Error verifying voice consistency: {consistency_metrics['error']}")
        return
    
    # Print summary
    print("\n=== Test Summary ===\n")
    print(f"Voice cloned: {voice_name}")
    print(f"Voice data directory: {voice_result['voice_dir']}")
    print(f"Synthesized speech: {synth_result['output_path']}")
    print(f"Prosody transfer: {prosody_result['output_path']}")
    print(f"Style transfer: {style_result['output_path']}")
    print(f"Voice consistency: {consistency_metrics['overall_similarity']:.2f} similarity score")
    
    print("\nNote: Since we're using placeholder implementations without real models,")
    print("the generated audio files contain silence or noise, and metrics are simulated.")
    print("In a real implementation with proper models, you would get actual voice output.")

if __name__ == "__main__":
    test_voice_cloning() 