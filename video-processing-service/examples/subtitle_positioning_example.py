#!/usr/bin/env python3
"""
Subtitle Positioning Example

This script demonstrates the subtitle positioning capabilities of the 
Subtitle Generation System, which intelligently places subtitles to avoid 
covering important visual content in videos.
"""

import sys
import logging
import asyncio
from pathlib import Path
import argparse
import os
import tempfile
import json

# Add parent directory to path to import from app
sys.path.append(str(Path(__file__).parent.parent))

from app.services.subtitles import (
    SubtitleService, SubtitleFormat, TextAlignment, TextPosition,
    SubtitleStyle
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sample transcript for demonstration
SAMPLE_TRANSCRIPT = {
    "segments": [
        {"start": 0.0, "end": 3.0, "text": "This is a subtitle positioned based on scene content."},
        {"start": 4.0, "end": 7.0, "text": "The system will analyze faces, objects, and text in the video."},
        {"start": 8.0, "end": 12.0, "text": "Subtitles will avoid covering important visual elements."},
        {"start": 13.0, "end": 16.0, "text": "If a face is detected, subtitles will be placed away from it."},
        {"start": 17.0, "end": 20.0, "text": "When important content is at the bottom, subtitles appear at the top."},
        {"start": 21.0, "end": 24.0, "text": "When important content is at the top, subtitles appear at the bottom."},
        {"start": 25.0, "end": 28.0, "text": "Scene composition analysis helps optimize viewer experience."}
    ]
}

async def analyze_video_for_positioning(video_path: str, output_dir: str, comparison_mode: bool = False):
    """
    Analyze a video for optimal subtitle positioning.
    
    Args:
        video_path: Path to the input video
        output_dir: Directory to save output files
        comparison_mode: Whether to generate both optimized and standard positioning for comparison
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize SubtitleService
    service = SubtitleService(
        config={
            "positioning_config": {
                "enable_face_detection": True,
                "enable_object_detection": True,
                "enable_text_detection": True,
                "frame_sample_rate": 5,  # Analyze more frames per second for demo
                "position_preference": ["bottom", "top", "center"]
            }
        }
    )
    
    # Get video file name
    video_file = os.path.basename(video_path)
    video_name = os.path.splitext(video_file)[0]
    
    # Analyze video for optimal subtitle positioning
    logger.info(f"Analyzing video for subtitle positioning: {video_path}")
    optimized_transcript = await service.optimize_subtitle_positioning(
        video_path=video_path,
        transcript=SAMPLE_TRANSCRIPT
    )
    
    # Save the optimized transcript to a JSON file
    optimized_json_path = output_path / f"{video_name}_optimized_transcript.json"
    with open(optimized_json_path, 'w') as f:
        json.dump(optimized_transcript, f, indent=2)
    
    logger.info(f"Saved optimized transcript to {optimized_json_path}")
    
    # Generate subtitle files
    formats = [SubtitleFormat.SRT, SubtitleFormat.VTT, SubtitleFormat.ASS]
    
    for format in formats:
        # Generate optimized subtitles
        optimized_subtitle_path = output_path / f"{video_name}_optimized.{format.value}"
        await service.generate_subtitles(
            transcript=optimized_transcript,
            output_path=str(optimized_subtitle_path),
            format=format
        )
        logger.info(f"Generated optimized {format.value.upper()} subtitles: {optimized_subtitle_path}")
        
        if comparison_mode:
            # Generate standard subtitles (always at bottom) for comparison
            standard_subtitle_path = output_path / f"{video_name}_standard.{format.value}"
            
            # Create a copy of the transcript with all subtitles at the bottom
            standard_transcript = {
                "segments": [
                    {**segment, "style": {"position": "bottom"}}
                    for segment in SAMPLE_TRANSCRIPT["segments"]
                ]
            }
            
            await service.generate_subtitles(
                transcript=standard_transcript,
                output_path=str(standard_subtitle_path),
                format=format
            )
            logger.info(f"Generated standard {format.value.upper()} subtitles: {standard_subtitle_path}")
    
    # Generate videos with subtitles if comparison mode is on
    if comparison_mode:
        # Generate video with optimized subtitle positioning
        optimized_video_path = output_path / f"{video_name}_optimized_subtitles.mp4"
        await service.render_video_with_subtitles(
            video_path=video_path,
            transcript=optimized_transcript,
            output_path=str(optimized_video_path)
        )
        logger.info(f"Generated video with optimized subtitle positioning: {optimized_video_path}")
        
        # Generate video with standard subtitle positioning
        standard_video_path = output_path / f"{video_name}_standard_subtitles.mp4"
        standard_transcript = {
            "segments": [
                {**segment, "style": {"position": "bottom"}}
                for segment in SAMPLE_TRANSCRIPT["segments"]
            ]
        }
        
        await service.render_video_with_subtitles(
            video_path=video_path,
            transcript=standard_transcript,
            output_path=str(standard_video_path)
        )
        logger.info(f"Generated video with standard subtitle positioning: {standard_video_path}")
    
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
    """Run the subtitle positioning example."""
    parser = argparse.ArgumentParser(description='Subtitle Positioning Example')
    parser.add_argument('--video', type=str, help='Path to input video file')
    parser.add_argument('--output', type=str, default='./output/subtitle_positioning', 
                      help='Output directory for generated files')
    parser.add_argument('--comparison', action='store_true', 
                      help='Generate comparison files with standard positioning')
    
    args = parser.parse_args()
    
    if not args.video:
        print("Error: Please provide a path to an input video file using the --video parameter.")
        print("Example: python subtitle_positioning_example.py --video path/to/video.mp4")
        return
    
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    print("Subtitle Positioning Example")
    print("============================")
    print(f"Input video: {args.video}")
    print(f"Output directory: {args.output}")
    print(f"Comparison mode: {'enabled' if args.comparison else 'disabled'}")
    
    # Run the positioning analysis
    optimized_transcript = await analyze_video_for_positioning(
        video_path=args.video,
        output_dir=args.output,
        comparison_mode=args.comparison
    )
    
    # Print the results
    print_positioning_results(optimized_transcript)
    
    print("\nExample completed successfully!")
    print(f"Output files have been saved to: {args.output}")

if __name__ == "__main__":
    asyncio.run(main()) 