#!/usr/bin/env python3
"""
Voice Animation Example Script

This example demonstrates how to use the voice-driven facial animation
capabilities of the avatar creation system.
"""

import os
import sys
import argparse
import time

# Add parent directory to path to help with imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.avatar_creation import (
    AvatarAnimator,
    VoiceAnimator,
    ensure_directory
)

def main():
    """
    Main function for the voice animation example.
    """
    parser = argparse.ArgumentParser(description="Voice-Driven Facial Animation Example")
    parser.add_argument('--avatar_path', type=str, required=True, 
                        help="Path to the avatar model file (.obj)")
    parser.add_argument('--audio_path', type=str, required=True, 
                        help="Path to the audio file")
    parser.add_argument('--output_dir', type=str, default="output/voice_animation", 
                        help="Directory to save animation output")
    parser.add_argument('--high_quality', action='store_true',
                        help="Use high-quality settings")
    parser.add_argument('--real_time', action='store_true',
                        help="Run in real-time streaming mode")
    parser.add_argument('--emotion_detection', action='store_true',
                        help="Enable emotion detection from voice")
    
    args = parser.parse_args()
    
    # Create output directory
    ensure_directory(args.output_dir)
    
    # Find related files
    avatar_dir = os.path.dirname(args.avatar_path)
    blendshapes_dir = os.path.join(avatar_dir, "blendshapes")
    skeleton_path = os.path.join(avatar_dir, "skeleton.json")
    texture_path = os.path.join(avatar_dir, "texture.png")
    
    print("=== Voice-Driven Facial Animation Example ===")
    print(f"Avatar: {args.avatar_path}")
    print(f"Audio: {args.audio_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Real-time mode: {args.real_time}")
    print(f"High quality: {args.high_quality}")
    print(f"Emotion detection: {args.emotion_detection}")
    print("===========================================")
    
    try:
        # Step 1: Create the avatar animator
        print("Step 1: Creating avatar animator")
        animator = AvatarAnimator(
            model_path=args.avatar_path,
            blendshapes_dir=blendshapes_dir if os.path.exists(blendshapes_dir) else None,
            skeleton_path=skeleton_path if os.path.exists(skeleton_path) else None,
            texture_path=texture_path if os.path.exists(texture_path) else None,
            use_gpu=True,
            high_quality=args.high_quality
        )
        
        available_blendshapes = animator.get_available_blendshapes()
        print(f"Avatar loaded with {len(available_blendshapes)} blendshapes")
        if available_blendshapes:
            print("Available blendshapes:")
            for i, bs in enumerate(available_blendshapes[:10]):  # Only show first 10
                print(f"  - {bs}")
            if len(available_blendshapes) > 10:
                print(f"  - ... and {len(available_blendshapes) - 10} more")
        
        # Step 2: Create the voice animator
        print("\nStep 2: Creating voice animator")
        voice_animator = VoiceAnimator(
            avatar_animator=animator,
            model_path=None,  # Use rule-based fallback
            use_gpu=True,
            smoothing_factor=0.3,
            emotion_detection=args.emotion_detection
        )
        
        # Step 3: Process the audio or start streaming
        print("\nStep 3: Processing animation")
        if args.real_time:
            print("Starting real-time animation (press Ctrl+C to stop)")
            
            # Start streaming from microphone
            voice_animator.start_streaming()
            
            try:
                # Keep the script running
                while True:
                    time.sleep(0.1)  # Small sleep to prevent high CPU usage
                    
                    # This would be where you'd handle additional real-time control
                    # e.g., keyboard input for changing expressions or poses
                    
            except KeyboardInterrupt:
                print("\nStopping streaming")
                voice_animator.stop_streaming()
            
        else:
            # Process audio file for lip-sync animation
            print(f"Processing audio file: {args.audio_path}")
            result = voice_animator.create_lipsync_animation(
                audio_path=args.audio_path,
                output_dir=args.output_dir,
                fps=30
            )
            
            # Print results
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print("\nAnimation completed successfully!")
                print(f"Created {result.get('frame_count', 0)} animation frames")
                print(f"Output saved to: {args.output_dir}")
                
                # Provide instructions for viewing/using the animation
                print("\nTo view the animation:")
                print("1. Use the frames in the output directory to create a video")
                print("2. For example: ffmpeg -framerate 30 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p animation.mp4")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 