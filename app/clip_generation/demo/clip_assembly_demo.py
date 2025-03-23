#!/usr/bin/env python3
"""
Clip Assembly and Optimization Demo

This script demonstrates the capabilities of the Clip Assembly Optimizer,
including smart clip generation, vertical format optimization, and multi-moment assembly.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import json
import time

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.insert(0, project_root)

# Import the necessary components
from app.clip_generation.services.clip_assembly import ClipAssemblyOptimizer, ClipAssemblyConfig
from app.clip_generation.services.clip_assembly_optimizer import AdvancedClipAssemblyOptimizer
from app.clip_generation.services.moment_detection import MomentAnalyzer, MomentAnalyzerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_moments(video_path, transcript_path=None):
    """
    Detect interesting moments in a video.
    
    Args:
        video_path: Path to the video file
        transcript_path: Optional path to a transcript file
        
    Returns:
        List of detected moments
    """
    logger.info(f"Analyzing video for interesting moments: {video_path}")
    
    # Create moment analyzer config
    config = MomentAnalyzerConfig(
        temp_dir="temp/moments",
        output_dir="output/moments",
        min_detection_score=0.6,
        enable_transcript_analysis=True
    )
    
    # Initialize analyzer
    analyzer = MomentAnalyzer(config)
    
    # Analyze the video
    moments = analyzer.analyze_video(video_path, transcript_path)
    
    # Convert moments to dictionary format
    moment_dicts = [
        {
            "start_time": moment.start_time,
            "end_time": moment.end_time,
            "combined_score": moment.combined_score,
            "duration": moment.end_time - moment.start_time,
            "scores": [
                {
                    "type": score.moment_type.value,
                    "score": score.score,
                    "confidence": score.confidence,
                    "metadata": score.metadata
                }
                for score in moment.scores
            ],
            "transcript": moment.transcript,
            "preview_image_path": moment.preview_image_path
        }
        for moment in moments
    ]
    
    logger.info(f"Detected {len(moment_dicts)} interesting moments")
    return moment_dicts


def demo_single_clip(
    optimizer, 
    video_path, 
    output_dir,
    vertical=False, 
    smart_endpoints=True,
    face_aware=False
):
    """
    Demonstrate generating a single optimized clip.
    
    Args:
        optimizer: ClipAssemblyOptimizer instance
        video_path: Path to the video file
        output_dir: Directory to save output clips
        vertical: Whether to generate a vertical format clip
        smart_endpoints: Whether to use smart endpoint detection
        face_aware: Whether to use face-aware cropping for vertical format
    """
    logger.info("=== Demo: Single Clip Generation ===")
    
    # Get video info to determine duration
    try:
        video_info = optimizer._get_video_info(video_path)
        video_duration = video_info["duration"]
        
        # Use a segment in the middle of the video
        start_time = video_duration * 0.4
        end_time = start_time + 15  # 15-second clip
        
        # Ensure end_time doesn't exceed video duration
        end_time = min(end_time, video_duration)
        
        logger.info(f"Selected clip range: {start_time:.2f}s - {end_time:.2f}s")
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        orientation = "vertical" if vertical else "horizontal"
        endpoints = "smart" if smart_endpoints else "fixed"
        face_suffix = "_face_aware" if face_aware and vertical else ""
        
        output_file = os.path.join(
            output_dir, 
            f"{base_name}_{orientation}_{endpoints}{face_suffix}.mp4"
        )
        
        # Generate the clip
        start_time_process = time.time()
        
        if vertical and face_aware and isinstance(optimizer, AdvancedClipAssemblyOptimizer):
            # Use face-aware vertical clip generation
            output_path = optimizer.generate_face_aware_vertical_clip(
                video_path,
                output_file,
                start_time,
                end_time,
                audio_normalize=True
            )
        else:
            # Use standard clip generation
            output_path = optimizer.generate_smart_clip(
                video_path,
                output_file,
                start_time,
                end_time,
                optimize_endpoints=smart_endpoints,
                vertical_format=vertical,
                audio_normalize=True
            )
        
        process_time = time.time() - start_time_process
        
        logger.info(f"Clip generated: {output_path}")
        logger.info(f"Processing time: {process_time:.2f} seconds")
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error generating single clip: {str(e)}")
        return None


def demo_multi_moment_assembly(
    optimizer, 
    video_path, 
    output_dir,
    vertical=False, 
    target_duration=30,
    face_aware=False
):
    """
    Demonstrate assembling multiple moments into a single clip.
    
    Args:
        optimizer: ClipAssemblyOptimizer instance
        video_path: Path to the video file
        output_dir: Directory to save output clips
        vertical: Whether to generate a vertical format clip
        target_duration: Target duration for the assembled clip
        face_aware: Whether to use face-aware cropping for vertical format
    """
    logger.info("=== Demo: Multi-Moment Clip Assembly ===")
    
    try:
        # First, detect interesting moments
        moments = detect_moments(video_path)
        
        if not moments:
            logger.warning("No interesting moments detected")
            return None
        
        logger.info(f"Using {len(moments)} detected moments")
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        orientation = "vertical" if vertical else "horizontal"
        face_suffix = "_face_aware" if face_aware and vertical else ""
        
        output_file = os.path.join(
            output_dir, 
            f"{base_name}_{orientation}_assembled_{target_duration}s{face_suffix}.mp4"
        )
        
        # Assemble the clip
        start_time_process = time.time()
        
        output_path = optimizer.assemble_multi_moment_clip(
            video_path,
            output_file,
            moments,
            vertical_format=vertical,
            target_duration=target_duration,
            optimize_transitions=True
        )
        
        process_time = time.time() - start_time_process
        
        logger.info(f"Assembled clip generated: {output_path}")
        logger.info(f"Processing time: {process_time:.2f} seconds")
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error assembling moments: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Clip Assembly and Optimization Demo")
    parser.add_argument("video_path", help="Path to the video file to process")
    parser.add_argument("--output-dir", "-o", default="output/clips", help="Directory to save output clips")
    parser.add_argument("--vertical", "-v", action="store_true", help="Generate vertical format clips")
    parser.add_argument("--assembly", "-a", action="store_true", help="Demonstrate multi-moment assembly")
    parser.add_argument("--duration", "-d", type=int, default=30, help="Target duration for assembled clip")
    parser.add_argument("--face-aware", "-f", action="store_true", help="Use face-aware cropping for vertical format")
    parser.add_argument("--advanced", action="store_true", help="Use advanced optimizations")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input
    if not os.path.exists(args.video_path):
        logger.error(f"Video file not found: {args.video_path}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create optimizer config
    config = ClipAssemblyConfig(
        temp_dir="temp/assembly",
        output_dir=str(output_dir),
        target_min_duration=10.0,
        target_max_duration=30.0
    )
    
    # Initialize optimizer
    if args.advanced or args.face_aware:
        logger.info("Using Advanced Clip Assembly Optimizer with face detection capabilities")
        optimizer = AdvancedClipAssemblyOptimizer(config)
    else:
        logger.info("Using standard Clip Assembly Optimizer")
        optimizer = ClipAssemblyOptimizer(config)
    
    # Run demonstration
    generated_clips = []
    
    # Generate single clips (both horizontal and vertical versions)
    single_clip = demo_single_clip(
        optimizer, 
        args.video_path, 
        str(output_dir),
        vertical=False, 
        smart_endpoints=True
    )
    if single_clip:
        generated_clips.append(single_clip)
    
    if args.vertical:
        # Standard vertical clip
        vertical_clip = demo_single_clip(
            optimizer, 
            args.video_path, 
            str(output_dir),
            vertical=True, 
            smart_endpoints=True,
            face_aware=False
        )
        if vertical_clip:
            generated_clips.append(vertical_clip)
            
        # Face-aware vertical clip (if requested and available)
        if args.face_aware and isinstance(optimizer, AdvancedClipAssemblyOptimizer):
            face_aware_clip = demo_single_clip(
                optimizer, 
                args.video_path, 
                str(output_dir),
                vertical=True, 
                smart_endpoints=True,
                face_aware=True
            )
            if face_aware_clip:
                generated_clips.append(face_aware_clip)
    
    # Demonstrate multi-moment assembly if requested
    if args.assembly:
        assembled_clip = demo_multi_moment_assembly(
            optimizer, 
            args.video_path, 
            str(output_dir),
            vertical=args.vertical, 
            target_duration=args.duration,
            face_aware=args.face_aware
        )
        if assembled_clip:
            generated_clips.append(assembled_clip)
    
    # Summary
    logger.info("\n=== Demo Summary ===")
    logger.info(f"Generated {len(generated_clips)} clips:")
    for clip in generated_clips:
        # Get clip duration
        try:
            clip_info = optimizer._get_video_info(clip)
            duration = clip_info["duration"]
            
            # Get file size
            size_mb = os.path.getsize(clip) / (1024 * 1024)
            
            logger.info(f"  - {os.path.basename(clip)}: {duration:.2f}s, {size_mb:.2f}MB")
        except Exception:
            logger.info(f"  - {os.path.basename(clip)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 