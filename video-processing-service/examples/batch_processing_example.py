#!/usr/bin/env python3
"""
Batch Processing Example Script

This script demonstrates how to use the BatchProcessor utility to process
multiple videos at once, generating subtitles in various formats and optionally
burning them into the videos.
"""

import os
import logging
import asyncio
import argparse
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Add the parent directory to the path so we can import from app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.utils.batch_processor import BatchProcessor
from app.services.subtitles import SubtitleService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Sample transcripts for demonstration
SAMPLE_TRANSCRIPTS = {
    "sample1": [
        {"start": 0.5, "end": 3.5, "text": "This is the first subtitle in our sample video."},
        {"start": 4.0, "end": 7.0, "text": "We are demonstrating batch processing capabilities."},
        {"start": 7.5, "end": 10.5, "text": "Multiple videos can be processed simultaneously."},
        {"start": 11.0, "end": 15.0, "text": "Each with its own transcript and output options."}
    ],
    "sample2": [
        {"start": 0.5, "end": 3.5, "text": "This is a different video with its own transcript."},
        {"start": 4.0, "end": 7.0, "text": "The batch processor handles all the videos efficiently."},
        {"start": 7.5, "end": 10.5, "text": "You can specify different formats for each video."},
        {"start": 11.0, "end": 15.0, "text": "And track progress through the callback system."}
    ]
}

class ProgressReporter:
    """Simple progress reporter for batch jobs."""
    
    def __init__(self):
        self.jobs = {}
        
    def report_progress(self, job_id: str, progress: float):
        """Report progress for a job."""
        self.jobs[job_id] = progress
        percent = int(progress * 100)
        bar_length = 30
        filled = int(bar_length * progress)
        bar = '█' * filled + '-' * (bar_length - filled)
        sys.stdout.write(f"\rJob {job_id}: [{bar}] {percent}%")
        sys.stdout.flush()
        
        if progress >= 1.0:
            print()  # New line after completion

async def process_directory(
    input_dir: str,
    output_dir: str,
    subtitle_formats: List[str] = ["srt", "vtt"],
    generate_video: bool = True,
    video_extensions: List[str] = [".mp4", ".avi", ".mov", ".mkv"]
) -> None:
    """
    Process all videos in a directory.
    
    Args:
        input_dir: Directory containing input videos
        output_dir: Directory to save outputs
        subtitle_formats: List of subtitle formats to generate
        generate_video: Whether to generate videos with burnt-in subtitles
        video_extensions: List of video file extensions to process
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the batch processor
    batch_processor = BatchProcessor(output_dir=output_dir)
    
    # Initialize progress reporter
    progress_reporter = ProgressReporter()
    
    # Collect video files
    video_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_paths.append(os.path.join(root, file))
    
    if not video_paths:
        logger.warning(f"No video files found in {input_dir}")
        return
    
    logger.info(f"Found {len(video_paths)} video files to process")
    
    # For demonstration, we'll use sample transcripts
    # In a real scenario, you would likely load actual transcripts for each video
    transcripts = []
    for i, video_path in enumerate(video_paths):
        # Alternate between sample transcripts for demonstration
        sample_key = "sample1" if i % 2 == 0 else "sample2"
        transcripts.append(SAMPLE_TRANSCRIPTS[sample_key])
    
    # Start batch processing
    job_id = await batch_processor.process_batch(
        video_paths=video_paths,
        transcripts=transcripts,
        subtitle_formats=subtitle_formats,
        generate_video=generate_video,
        progress_callback=progress_reporter.report_progress,
        batch_options={
            "video_quality": "medium",
            "optimize_positioning": True,
            "style_name": "default",
            "reading_speed_preset": "normal",
            "detect_emphasis": True,
            "auto_detect_language": True
        }
    )
    
    logger.info(f"Started batch job {job_id}")
    
    # Wait for job to complete (for demonstration)
    while True:
        job_status = batch_processor.get_job_status(job_id)
        if job_status["status"] == "completed":
            break
        await asyncio.sleep(1)
    
    # Print job summary
    job_dir = os.path.join(output_dir, job_id)
    summary_path = os.path.join(job_dir, "batch_summary.json")
    
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        logger.info("Batch processing completed:")
        logger.info(f"Total videos: {summary['total_videos']}")
        logger.info(f"Completed: {summary['completed_videos']}")
        logger.info(f"Failed: {summary['failed_videos']}")
        logger.info(f"Output directory: {job_dir}")
    else:
        logger.warning("Summary file not found")

async def run_single_batch_example(
    output_dir: str
) -> None:
    """
    Run a simple batch processing example with predefined videos.
    
    Args:
        output_dir: Directory to save outputs
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the batch processor
    batch_processor = BatchProcessor(output_dir=output_dir)
    
    # Initialize progress reporter
    progress_reporter = ProgressReporter()
    
    # For demonstration, we'll use sample videos
    # In a real scenario, you would use actual video paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    samples_dir = os.path.join(Path(current_dir).parent, "samples")
    
    # Check if samples directory exists
    if not os.path.exists(samples_dir):
        logger.warning(f"Samples directory not found: {samples_dir}")
        # Create a test samples directory with placeholder files
        os.makedirs(samples_dir, exist_ok=True)
        open(os.path.join(samples_dir, "sample_video1.mp4"), 'w').close()
        open(os.path.join(samples_dir, "sample_video2.mp4"), 'w').close()
    
    # Collect sample videos
    video_paths = []
    for file in os.listdir(samples_dir):
        if file.endswith(".mp4"):
            video_paths.append(os.path.join(samples_dir, file))
    
    if not video_paths:
        logger.warning(f"No sample videos found in {samples_dir}")
        # Create placeholder video paths for demonstration
        video_paths = [
            os.path.join(samples_dir, "sample_video1.mp4"),
            os.path.join(samples_dir, "sample_video2.mp4")
        ]
    
    # Use sample transcripts for demonstration
    transcripts = [
        SAMPLE_TRANSCRIPTS["sample1"],
        SAMPLE_TRANSCRIPTS["sample2"]
    ]
    
    # Start batch processing
    job_id = await batch_processor.process_batch(
        video_paths=video_paths,
        transcripts=transcripts,
        subtitle_formats=["srt", "vtt", "ass"],
        generate_video=True,
        progress_callback=progress_reporter.report_progress,
        batch_options={
            "video_quality": "medium",
            "optimize_positioning": True,
            "style_name": "default",
            "reading_speed_preset": "normal",
            "detect_emphasis": True,
            "auto_detect_language": True
        }
    )
    
    logger.info(f"Started batch job {job_id}")
    
    # Monitor job status
    while True:
        job_status = batch_processor.get_job_status(job_id)
        completed = job_status["completed_videos"]
        failed = job_status["failed_videos"]
        total = job_status["total_videos"]
        
        if job_status["status"] == "completed":
            logger.info(f"Batch job completed: {completed}/{total} succeeded, {failed}/{total} failed")
            break
        
        logger.info(f"Progress: {completed + failed}/{total} videos processed")
        await asyncio.sleep(2)
    
    # Print job summary
    job_dir = os.path.join(output_dir, job_id)
    summary_path = os.path.join(job_dir, "batch_summary.json")
    
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        logger.info("\nBatch Processing Results:")
        logger.info("-" * 50)
        logger.info(f"Total videos: {summary['total_videos']}")
        logger.info(f"Completed: {summary['completed_videos']}")
        logger.info(f"Failed: {summary['failed_videos']}")
        logger.info(f"Output directory: {job_dir}")
        logger.info("-" * 50)
        
        # Print details of each video
        for i, result in enumerate(summary['results']):
            logger.info(f"\nVideo {i+1}: {os.path.basename(result['video_path'])}")
            logger.info(f"Status: {result['status']}")
            
            if result['status'] == 'success':
                logger.info("Generated files:")
                for file_type, file_path in result.get('files', {}).items():
                    logger.info(f"  - {file_type}: {os.path.basename(file_path)}")
            elif result['status'] == 'failed':
                logger.info(f"Error: {result.get('error', 'Unknown error')}")
    else:
        logger.warning("Summary file not found")

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Batch Processing Example')
    parser.add_argument('--input-dir', type=str, help='Input directory containing videos')
    parser.add_argument('--output-dir', type=str, default='./batch_output', help='Output directory for results')
    parser.add_argument('--formats', type=str, default='srt,vtt', help='Comma-separated list of subtitle formats')
    parser.add_argument('--video', action='store_true', help='Generate videos with burnt-in subtitles')
    parser.add_argument('--example', action='store_true', help='Run the built-in example instead of processing a directory')
    
    args = parser.parse_args()
    
    # Parse subtitle formats
    subtitle_formats = args.formats.split(',')
    
    if args.example:
        # Run the built-in example
        await run_single_batch_example(args.output_dir)
    elif args.input_dir:
        # Process the specified directory
        await process_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            subtitle_formats=subtitle_formats,
            generate_video=args.video
        )
    else:
        logger.error("Either --input-dir or --example must be specified")
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main()) 