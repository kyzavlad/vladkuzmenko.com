#!/usr/bin/env python3
"""
Silence Detection Demo Script

This script demonstrates the Silent/Unnecessary Audio Detection system
for the Clip Generation Microservice, allowing users to process videos
by removing or speeding up silent parts and unnecessary sounds.
"""

import os
import sys
import argparse
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add parent directory to path to allow importing from the parent package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from app.clip_generation.services.audio_analysis.silence_processor import SilenceProcessor, SilenceProcessorConfig
from app.clip_generation.services.audio_analysis.silence_detector import SilenceDetectorConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Silence Detection Demo Script for Clip Generation Microservice'
    )
    
    # Input and output options
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input video file or directory containing videos')
    parser.add_argument('--output', '-o', type=str, default='output',
                       help='Output directory for processed videos')
    
    # Processing options
    parser.add_argument('--mode', '-m', type=str, choices=['remove', 'speedup'], default='remove',
                       help='Processing mode: remove silence or speed it up')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                       help='Minimum duration threshold for removal (seconds)')
    parser.add_argument('--max-gap', '-g', type=float, default=0.1,
                       help='Maximum gap between segments to merge (seconds)')
    parser.add_argument('--speed-factor', '-s', type=float, default=2.0,
                       help='Speed factor for silence when using speedup mode')
    
    # Audio analysis options
    parser.add_argument('--min-silence', type=float, default=0.3,
                       help='Minimum silence duration to detect (seconds)')
    parser.add_argument('--max-silence', type=float, default=2.0,
                       help='Maximum silence duration to keep (seconds)')
    parser.add_argument('--silence-threshold', type=float, default=-35.0,
                       help='dB threshold for silence detection')
    parser.add_argument('--adaptive-threshold', action='store_true',
                       help='Use adaptive threshold based on noise profile')
    
    # Filler word detection options
    parser.add_argument('--detect-fillers', action='store_true',
                       help='Enable filler word detection')
    parser.add_argument('--language', type=str, default='en',
                       help='Language for filler word detection')
    
    # Performance options
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel processing for batch mode')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of worker threads for parallel processing')
    
    # Advanced options
    parser.add_argument('--preserve-quality', action='store_true',
                       help='Preserve original video quality (uses more disk space)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations of the analysis')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu',
                       help='Device for ML inference (cuda requires GPU)')
    
    return parser.parse_args()


def create_configs(args):
    """Create configuration objects from command line arguments."""
    # Create silence detector configuration
    detector_config = SilenceDetectorConfig(
        min_silence_duration=args.min_silence,
        max_silence_duration=args.max_silence,
        silence_threshold=args.silence_threshold,
        adaptive_threshold=args.adaptive_threshold,
        visualize=args.visualize,
        enable_filler_detection=args.detect_fillers,
        language=args.language,
        device=args.device
    )
    
    # Create silence processor configuration
    processor_config = SilenceProcessorConfig(
        output_dir=args.output,
        removal_threshold=args.threshold,
        max_segment_gap=args.max_gap,
        speed_up_silence=(args.mode == 'speedup'),
        speed_factor=args.speed_factor,
        parallel_processing=args.parallel,
        max_workers=args.workers,
        preserve_video_quality=args.preserve_quality,
        silence_detector_config=detector_config
    )
    
    return detector_config, processor_config


def process_inputs(processor, input_path):
    """Process input files or directories."""
    input_path = Path(input_path)
    
    if input_path.is_file():
        # Process single file
        logger.info(f"Processing single file: {input_path}")
        result = processor.process_video(str(input_path))
        
        if result:
            logger.info(f"Processing complete. Output saved to: {result}")
            return True
        else:
            logger.error("Processing failed.")
            return False
            
    elif input_path.is_dir():
        # Process directory (batch mode)
        logger.info(f"Processing directory: {input_path}")
        
        # Find video files
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
        video_paths = []
        
        for ext in video_extensions:
            video_paths.extend(list(input_path.glob(f'*{ext}')))
        
        if not video_paths:
            logger.error(f"No video files found in {input_path}")
            return False
        
        logger.info(f"Found {len(video_paths)} video files")
        
        # Process videos in batch
        report = processor.batch_process([str(p) for p in video_paths])
        
        # Display summary
        summary = report["summary"]
        logger.info(f"Batch processing summary:")
        logger.info(f"  Processed {summary['processed_files']}/{summary['total_files']} files")
        logger.info(f"  Success rate: {summary['success_rate']:.1f}%")
        logger.info(f"  Total duration: {summary['total_duration']:.2f}s")
        logger.info(f"  Removed duration: {summary['removed_duration']:.2f}s")
        logger.info(f"  Reduction: {summary['reduction_percentage']:.1f}%")
        logger.info(f"  Processing time: {summary['processing_time']:.2f}s")
        
        return summary['processed_files'] > 0
        
    else:
        logger.error(f"Input path does not exist: {input_path}")
        return False


def main():
    """Main entry point for the script."""
    start_time = time.time()
    
    # Parse command line arguments
    args = parse_args()
    
    # Create configurations
    detector_config, processor_config = create_configs(args)
    
    # Initialize the silence processor
    processor = SilenceProcessor(processor_config)
    
    # Process inputs
    success = process_inputs(processor, args.input)
    
    # Display final statistics
    stats = processor.get_stats()
    logger.info(f"Total processed files: {stats['processed_files']}")
    
    if stats['processed_files'] > 0:
        logger.info(f"Average reduction: {stats['reduction_percentage']:.1f}%")
    
    logger.info(f"Total execution time: {time.time() - start_time:.2f}s")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 