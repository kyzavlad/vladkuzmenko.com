#!/usr/bin/env python3
"""
Lightweight Subtitle Positioning Example

This script demonstrates the lightweight subtitle positioning capabilities of the 
Subtitle Generation System, which places subtitles to avoid covering faces and
important visual content while using minimal system resources.
"""

import sys
import logging
import asyncio
from pathlib import Path
import argparse
import os
import json
import time

# Add parent directory to path to import from app
sys.path.append(str(Path(__file__).parent.parent))

from app.services.subtitles import (
    SubtitleFormat, TextAlignment, TextPosition
)
from app.services.subtitles.subtitle_positioning_lite import SubtitlePositioningLite

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sample transcript for demonstration
SAMPLE_TRANSCRIPT = {
    "segments": [
        {"start": 0.0, "end": 3.0, "text": "This is a simple subtitle positioned to avoid faces."},
        {"start": 4.0, "end": 7.0, "text": "The lightweight system analyzes basic visual elements."},
        {"start": 8.0, "end": 12.0, "text": "Subtitles will move to avoid covering important content."},
        {"start": 13.0, "end": 16.0, "text": "When faces are detected at the bottom, subtitles move to the top."},
        {"start": 17.0, "end": 20.0, "text": "When faces are detected at the top, subtitles stay at the bottom."},
        {"start": 21.0, "end": 24.0, "text": "This is optimized for speed and minimal resource usage."}
    ]
}

async def analyze_video_with_lite_positioning(video_path: str, output_dir: str):
    """
    Analyze a video using the lightweight subtitle positioning.
    
    Args:
        video_path: Path to the input video
        output_dir: Directory to save output files
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize positioning service
    positioning = SubtitlePositioningLite(
        config={
            "enable_face_detection": True,
            "enable_complexity_analysis": True,
            "frame_sample_rate": 24,  # Analyze 1 frame per second (for 24fps video)
            "min_frames": 10,
            "max_frames": 50,
            "position_preference": ["bottom", "top"]
        }
    )
    
    # Get video file name
    video_file = os.path.basename(video_path)
    video_name = os.path.splitext(video_file)[0]
    
    # Measure execution time
    start_time = time.time()
    
    # Analyze video for optimal subtitle positioning
    logger.info(f"Analyzing video for subtitle positioning: {video_path}")
    optimized_transcript = await positioning.analyze_video_for_positioning(
        video_path=video_path,
        transcript=SAMPLE_TRANSCRIPT
    )
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Save the optimized transcript to a JSON file
    optimized_json_path = output_path / f"{video_name}_optimized_transcript.json"
    with open(optimized_json_path, 'w') as f:
        json.dump(optimized_transcript, f, indent=2)
    
    logger.info(f"Saved optimized transcript to {optimized_json_path}")
    
    # Print statistics
    print(f"\nExecution time: {execution_time:.2f} seconds")
    
    # Count segments with different positions
    positions = {"top": 0, "bottom": 0, "middle": 0, "other": 0}
    
    for segment in optimized_transcript.get("segments", []):
        if "style" in segment and "position" in segment["style"]:
            position = segment["style"]["position"]
            if position in positions:
                positions[position] += 1
            else:
                positions["other"] += 1
        else:
            positions["bottom"] += 1  # Default position
    
    print("\nPositioning statistics:")
    for position, count in positions.items():
        if count > 0:
            print(f"  {position.capitalize()}: {count} segments")
    
    return optimized_transcript

def print_positioning_results(transcript: dict):
    """Print the positioning results for each subtitle segment."""
    print("\nSUBTITLE POSITIONING RESULTS")
    print("===========================")
    
    for i, segment in enumerate(transcript.get("segments", [])):
        position = "bottom"  # Default position
        if "style" in segment and "position" in segment["style"]:
            position = segment["style"]["position"]
        
        # Get timing information
        start = segment.get("start", 0)
        end = segment.get("end", 0)
        duration = end - start
        
        print(f"\nSegment {i+1} ({start:.1f}s - {end:.1f}s, duration: {duration:.1f}s)")
        print(f"Text: {segment.get('text', '')}")
        print(f"Position: {position.upper()}")

async def main():
    """Run the lightweight subtitle positioning example."""
    parser = argparse.ArgumentParser(description='Lightweight Subtitle Positioning Example')
    parser.add_argument('--video', type=str, help='Path to input video file')
    parser.add_argument('--output', type=str, default='./output/subtitle_positioning_lite', 
                      help='Output directory for generated files')
    
    args = parser.parse_args()
    
    if not args.video:
        print("Error: Please provide a path to an input video file using the --video parameter.")
        print("Example: python subtitle_positioning_lite_example.py --video path/to/video.mp4")
        return
    
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    print("Lightweight Subtitle Positioning Example")
    print("=======================================")
    print(f"Input video: {args.video}")
    print(f"Output directory: {args.output}")
    
    # Run the positioning analysis
    optimized_transcript = await analyze_video_with_lite_positioning(
        video_path=args.video,
        output_dir=args.output
    )
    
    # Print the results
    print_positioning_results(optimized_transcript)
    
    print("\nExample completed successfully!")
    print(f"Output files have been saved to: {args.output}")

if __name__ == "__main__":
    asyncio.run(main()) 