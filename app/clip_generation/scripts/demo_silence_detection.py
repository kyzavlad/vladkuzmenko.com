#!/usr/bin/env python3
"""
Demonstration script for the Clip Generation Service with silence detection.

This script provides a command-line interface to generate clips from videos
with optional silence detection and removal.
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from app.clip_generation.services.clip_generation_service import ClipGenerationService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate clips from videos with optional silence detection and removal"
    )
    
    parser.add_argument(
        "source_video",
        help="Path to the source video file"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Path to output clip file (default: auto-generated)",
        default=None
    )
    
    parser.add_argument(
        "-s", "--start",
        help="Start time for the clip in seconds or HH:MM:SS format",
        required=True
    )
    
    parser.add_argument(
        "-e", "--end",
        help="End time for the clip in seconds or HH:MM:SS format",
        required=True
    )
    
    parser.add_argument(
        "--silence-detection",
        help="Enable silence detection and removal",
        action="store_true"
    )
    
    parser.add_argument(
        "--min-silence",
        help="Minimum silence duration in seconds (default: 0.3)",
        type=float,
        default=0.3
    )
    
    parser.add_argument(
        "--max-silence",
        help="Maximum silence duration in seconds (default: 2.0)",
        type=float,
        default=2.0
    )
    
    parser.add_argument(
        "--threshold",
        help="Silence threshold in dB (default: -35.0)",
        type=float,
        default=-35.0
    )
    
    parser.add_argument(
        "--adaptive",
        help="Use adaptive threshold based on audio level",
        action="store_true"
    )
    
    parser.add_argument(
        "--detect-fillers",
        help="Enable filler word detection (um, uh, etc.)",
        action="store_true"
    )
    
    parser.add_argument(
        "--speed-up",
        help="Speed up silent parts instead of removing them",
        action="store_true"
    )
    
    parser.add_argument(
        "--speed-factor",
        help="Speed factor for silent parts (default: 2.0)",
        type=float,
        default=2.0
    )
    
    parser.add_argument(
        "--output-dir",
        help="Directory for output files (default: ./output)",
        default="./output"
    )
    
    parser.add_argument(
        "--temp-dir",
        help="Directory for temporary files (default: auto-generated)",
        default=None
    )
    
    parser.add_argument(
        "--ffmpeg-path",
        help="Path to ffmpeg binary (default: ffmpeg in PATH)",
        default="ffmpeg"
    )
    
    parser.add_argument(
        "--verbose",
        help="Enable verbose logging",
        action="store_true"
    )
    
    return parser.parse_args()


def parse_time_format(time_str):
    """Parse time string in seconds or HH:MM:SS format."""
    if ":" in time_str:
        # Parse HH:MM:SS format
        parts = time_str.split(":")
        if len(parts) == 3:
            # HH:MM:SS
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        elif len(parts) == 2:
            # MM:SS
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
    
    # Parse seconds format
    return float(time_str)


def status_callback(status_update: Dict[str, Any]):
    """Callback function for status updates."""
    status = status_update["status"]
    progress = status_update["progress"]
    message = status_update["message"]
    
    # Print progress bar
    bar_length = 40
    filled_length = int(bar_length * progress / 100)
    bar = "█" * filled_length + "░" * (bar_length - filled_length)
    
    print(f"\r[{bar}] {progress:.1f}% - {message}", end="")
    
    if status in ["completed", "error"]:
        print()  # Add newline for completed or error status


def main():
    """Main function to run the demo."""
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse clip start and end times
    try:
        clip_start = parse_time_format(args.start)
        clip_end = parse_time_format(args.end)
    except ValueError:
        logger.error("Invalid time format. Use seconds or HH:MM:SS format")
        return 1
    
    # Check if source video exists
    if not os.path.exists(args.source_video):
        logger.error(f"Source video not found: {args.source_video}")
        return 1
    
    # Configure silence detection
    silence_detection_config = {
        "min_silence": args.min_silence,
        "max_silence": args.max_silence,
        "threshold": args.threshold,
        "adaptive_threshold": args.adaptive,
        "detect_fillers": args.detect_fillers,
        "speed_up": args.speed_up,
        "speed_factor": args.speed_factor,
    }
    
    # Initialize the clip generation service
    service = ClipGenerationService(
        output_dir=args.output_dir,
        temp_dir=args.temp_dir,
        ffmpeg_path=args.ffmpeg_path,
        enable_silence_detection=True,  # Always enable, but only use if --silence-detection is specified
        silence_detection_config=silence_detection_config
    )
    
    # Create task
    task = {
        "task_id": f"demo_{int(time.time())}",
        "source_video": args.source_video,
        "clip_start": clip_start,
        "clip_end": clip_end,
        "output_file": args.output,
        "remove_silence": args.silence_detection,
        "silence_config": silence_detection_config
    }
    
    print(f"Processing video: {args.source_video}")
    print(f"Clip time: {clip_start:.2f}s to {clip_end:.2f}s (duration: {clip_end - clip_start:.2f}s)")
    print(f"Silence detection: {'Enabled' if args.silence_detection else 'Disabled'}")
    if args.silence_detection:
        print(f"Silence settings:")
        print(f"  - Min silence duration: {args.min_silence}s")
        print(f"  - Max silence duration: {args.max_silence}s")
        print(f"  - Threshold: {args.threshold} dB")
        print(f"  - Adaptive threshold: {'Enabled' if args.adaptive else 'Disabled'}")
        print(f"  - Filler word detection: {'Enabled' if args.detect_fillers else 'Disabled'}")
        print(f"  - Speed up silent parts: {'Enabled' if args.speed_up else 'Disabled'}")
        if args.speed_up:
            print(f"  - Speed factor: {args.speed_factor}x")
    print()
    print("Starting processing...")
    
    # Process the task
    start_time = time.time()
    result = service.process_clip_task(task, status_callback)
    end_time = time.time()
    
    if result["status"] == "error":
        print(f"\nError: {result['error']}")
        return 1
    
    # Print results
    print(f"\nProcessing completed in {end_time - start_time:.2f} seconds")
    print(f"Output file: {result['output_file']}")
    
    if args.silence_detection and "silence_stats" in result:
        stats = result["silence_stats"]
        print("\nSilence detection statistics:")
        print(f"  - Original duration: {stats['original_duration']:.2f}s")
        print(f"  - Processed duration: {stats['processed_duration']:.2f}s")
        print(f"  - Reduction: {stats['reduction_percentage']:.1f}%")
        print(f"  - Segments detected: {stats['segments_detected']}")
        print(f"  - Segments removed: {stats['segments_removed']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 