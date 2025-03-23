#!/usr/bin/env python3
"""
Dynamic Volume Adjustment Example

This script demonstrates the usage of the VolumeAdjuster class to automatically
adjust music volume during speech segments in a video.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.music.volume_adjuster import VolumeAdjuster

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to demonstrate dynamic volume adjustment."""
    parser = argparse.ArgumentParser(description="Dynamic volume adjustment example")
    parser.add_argument('video', type=str, help='Path to input video file')
    parser.add_argument('music', type=str, help='Path to music file')
    parser.add_argument('output', type=str, help='Path to output video file')
    parser.add_argument('--start', type=float, default=0.0, help='Start time for music (seconds)')
    parser.add_argument('--end', type=float, default=None, help='End time for music (seconds, default: end of video)')
    parser.add_argument('--default-volume', type=float, default=0.7, help='Default music volume (0.0-1.0)')
    parser.add_argument('--ducking-amount', type=float, default=0.3, help='Volume during speech (0.0-1.0)')
    parser.add_argument('--fade-in', type=float, default=0.5, help='Fade-in time after speech (seconds)')
    parser.add_argument('--fade-out', type=float, default=0.8, help='Fade-out time before speech (seconds)')
    parser.add_argument('--no-original-audio', action='store_true', help='Replace original audio instead of mixing')
    parser.add_argument('--speech-threshold', type=float, default=-25, help='Speech detection threshold (dB)')
    parser.add_argument('--speech-min-duration', type=float, default=0.3, help='Minimum speech duration (seconds)')
    parser.add_argument('--ffmpeg', type=str, default='ffmpeg', help='Path to ffmpeg')
    parser.add_argument('--ffprobe', type=str, default='ffprobe', help='Path to ffprobe')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.video):
        logger.error(f"Video file not found: {args.video}")
        return 1
        
    if not os.path.exists(args.music):
        logger.error(f"Music file not found: {args.music}")
        return 1
    
    # Make sure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Configure volume adjuster
    config = {
        'ffmpeg_path': args.ffmpeg,
        'ffprobe_path': args.ffprobe,
        'default_music_volume': args.default_volume,
        'ducking_amount': args.ducking_amount,
        'fade_in_time': args.fade_in,
        'fade_out_time': args.fade_out,
        'speech_detection_threshold': args.speech_threshold,
        'speech_min_duration': args.speech_min_duration
    }
    
    adjuster = VolumeAdjuster(config)
    
    logger.info(f"Processing video: {args.video}")
    logger.info(f"Adding music: {args.music}")
    logger.info(f"Output will be saved to: {args.output}")
    
    # Process video with dynamic volume adjustment
    result = adjuster.adjust_music_for_speech(
        video_path=args.video,
        music_path=args.music,
        output_path=args.output,
        music_start_time=args.start,
        music_end_time=args.end,
        keep_original_audio=not args.no_original_audio
    )
    
    # Check result
    if result['status'] == 'success':
        logger.info(f"✅ Success! Output saved to: {result['output_path']}")
        
        # Print speech segments info
        speech_segments = result.get('speech_segments', [])
        if speech_segments:
            logger.info(f"Detected {len(speech_segments)} speech segments:")
            for i, segment in enumerate(speech_segments, 1):
                duration = segment['end'] - segment['start']
                logger.info(f"  Segment {i}: {segment['start']:.2f}s - {segment['end']:.2f}s (duration: {duration:.2f}s)")
        
        # Optionally save detailed results
        details_path = f"{os.path.splitext(args.output)[0]}_details.json"
        with open(details_path, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Detailed results saved to: {details_path}")
        
        return 0
    else:
        logger.error(f"❌ Error: {result.get('error', 'Unknown error')}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 