#!/usr/bin/env python3
"""
Multiple Subtitle Output Example

This script demonstrates generating multiple subtitle outputs from a single video:
- Burnt-in subtitles directly in the video
- Separate subtitle files in various formats (SRT, VTT, ASS)
- All outputs organized with a JSON manifest
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
    SubtitleService, SubtitleFormat, TextAlignment, TextPosition
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
        {"start": 0.0, "end": 3.0, "text": "This is a sample subtitle for demonstration."},
        {"start": 4.0, "end": 7.0, "text": "We're creating multiple output formats at once."},
        {"start": 8.0, "end": 12.0, "text": "Including burnt-in subtitles directly in the video."},
        {"start": 13.0, "end": 16.0, "text": "As well as separate SRT, VTT, and ASS files."},
        {"start": 17.0, "end": 20.0, "text": "This makes distribution much easier!"},
        {"start": 21.0, "end": 25.0, "text": "You can choose which formats work best for your needs."}
    ]
}

class ProgressReporter:
    """Simple progress reporter for demonstration."""
    
    def __init__(self):
        self.last_update = 0
    
    def report_progress(self, progress: float):
        """Report progress percentage."""
        # Only update if significant progress has been made (avoid console spam)
        if progress - self.last_update >= 0.05 or progress >= 1.0:
            print(f"Progress: {progress*100:.1f}%")
            self.last_update = progress

async def generate_multiple_outputs(video_path: str, output_dir: str, formats: list):
    """
    Generate multiple subtitle outputs for a video.
    
    Args:
        video_path: Path to the input video
        output_dir: Directory to save output files
        formats: List of subtitle formats to generate
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize SubtitleService
    service = SubtitleService()
    
    # Create a progress reporter
    progress_reporter = ProgressReporter()
    
    # Measure execution time
    start_time = time.time()
    
    # Generate multiple outputs
    logger.info(f"Generating multiple subtitle outputs for: {video_path}")
    
    results = await service.generate_multiple_outputs(
        video_path=video_path,
        transcript=SAMPLE_TRANSCRIPT,
        output_dir=str(output_path),
        subtitle_formats=formats,
        generate_video=True,
        video_quality="medium",
        optimize_positioning=True,
        detect_emphasis=True,
        auto_detect_language=True,
        show_progress_callback=progress_reporter.report_progress
    )
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Print the results
    print("\n===== GENERATION RESULTS =====")
    print(f"Execution time: {execution_time:.2f} seconds")
    print("\nGenerated files:")
    
    for output_type, file_path in results.items():
        if output_type != "manifest":
            file_size = os.path.getsize(file_path) / 1024  # Size in KB
            print(f"  {output_type}: {os.path.basename(file_path)} ({file_size:.1f} KB)")
    
    # Load and display the manifest
    with open(results.get("manifest", ""), "r") as f:
        manifest = json.load(f)
    
    print("\nManifest information:")
    print(f"  Source video: {os.path.basename(manifest.get('source_video', ''))}")
    print(f"  Generated at: {manifest.get('generated_at', '')}")
    print(f"  Options: {json.dumps(manifest.get('subtitle_options', {}), indent=2)}")
    
    print("\nExample completed successfully!")
    print(f"All output files have been saved to: {output_dir}")

async def main():
    """Run the multiple subtitle output example."""
    parser = argparse.ArgumentParser(description='Multiple Subtitle Output Example')
    parser.add_argument('--video', type=str, help='Path to input video file')
    parser.add_argument('--output', type=str, default='./output/multiple_output', 
                      help='Output directory for generated files')
    parser.add_argument('--formats', type=str, default='srt,vtt,ass',
                      help='Comma-separated list of subtitle formats to generate')
    
    args = parser.parse_args()
    
    if not args.video:
        print("Error: Please provide a path to an input video file using the --video parameter.")
        print("Example: python multiple_output_example.py --video path/to/video.mp4")
        return
    
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    # Parse formats
    formats = [fmt.strip().lower() for fmt in args.formats.split(',')]
    
    print("Multiple Subtitle Output Example")
    print("================================")
    print(f"Input video: {args.video}")
    print(f"Output directory: {args.output}")
    print(f"Subtitle formats: {', '.join(formats)}")
    
    # Generate the multiple outputs
    await generate_multiple_outputs(
        video_path=args.video,
        output_dir=args.output,
        formats=formats
    )

if __name__ == "__main__":
    asyncio.run(main()) 