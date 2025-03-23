#!/usr/bin/env python3
"""
Interesting Moment Detection Demo

This script demonstrates the capabilities of the Interesting Moment Detection system,
showing how to analyze videos to find engaging moments and generate highlight clips.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.insert(0, project_root)

# Import the moment detection components
from app.clip_generation.services.moment_detection import (
    MomentAnalyzer,
    MomentAnalyzerConfig,
    ContentAnalyzer,
    TranscriptAnalyzer,
    VoiceAnalyzer,
    MomentType,
    MomentScore
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def visualize_moments(moments, video_duration, output_path=None):
    """
    Visualize the detected moments and their scores.
    
    Args:
        moments: List of DetectedMoment objects
        video_duration: Duration of the video in seconds
        output_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Sort moments by start time
    moments = sorted(moments, key=lambda m: m.start_time)
    
    # Create a timeline
    timeline = np.zeros(int(video_duration * 10))  # Resolution: 10 points per second
    
    # Prepare colors for different moment types
    colors = {
        MomentType.AUDIO_PEAK: 'red',
        MomentType.VOICE_EMPHASIS: 'orange',
        MomentType.LAUGHTER: 'pink',
        MomentType.REACTION: 'purple',
        MomentType.SENTIMENT_PEAK: 'blue',
        MomentType.KEYWORD: 'green',
        MomentType.KEY_POINT: 'cyan'
    }
    
    # Plot each moment type separately
    moment_types = set(score.moment_type for moment in moments for score in moment.scores)
    
    # Plot individual moment scores
    for i, moment_type in enumerate(moment_types):
        type_timeline = np.zeros(len(timeline))
        
        for moment in moments:
            for score in moment.scores:
                if score.moment_type == moment_type:
                    start_idx = int(moment.start_time * 10)
                    end_idx = int(moment.end_time * 10)
                    if end_idx >= len(type_timeline):
                        end_idx = len(type_timeline) - 1
                    type_timeline[start_idx:end_idx] = max(
                        type_timeline[start_idx:end_idx],
                        np.ones(end_idx - start_idx) * score.score
                    )
        
        plt.plot(
            np.arange(len(type_timeline)) / 10,
            type_timeline,
            label=moment_type.value,
            color=colors.get(moment_type, f'C{i}'),
            alpha=0.6
        )
    
    # Plot combined scores
    combined_timeline = np.zeros(len(timeline))
    for moment in moments:
        start_idx = int(moment.start_time * 10)
        end_idx = int(moment.end_time * 10)
        if end_idx >= len(combined_timeline):
            end_idx = len(combined_timeline) - 1
        combined_timeline[start_idx:end_idx] = max(
            combined_timeline[start_idx:end_idx],
            np.ones(end_idx - start_idx) * moment.combined_score
        )
    
    plt.plot(
        np.arange(len(combined_timeline)) / 10,
        combined_timeline,
        label='Combined Score',
        color='black',
        linewidth=2
    )
    
    # Highlight the top moments
    top_moments = sorted(moments, key=lambda m: m.combined_score, reverse=True)[:5]
    for i, moment in enumerate(top_moments):
        plt.axvspan(
            moment.start_time,
            moment.end_time,
            alpha=0.3,
            color='yellow',
            label=f'Top Moment {i+1}' if i == 0 else None
        )
        plt.text(
            (moment.start_time + moment.end_time) / 2,
            0.95,
            f'{i+1}',
            horizontalalignment='center',
            verticalalignment='center',
            bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8)
        )
    
    # Set plot details
    plt.xlabel('Time (seconds)')
    plt.ylabel('Score')
    plt.title('Interesting Moments Detection Timeline')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {output_path}")
    else:
        plt.show()


def summarize_moments(moments):
    """
    Print a summary of detected moments.
    
    Args:
        moments: List of DetectedMoment objects
    """
    print("\n======= DETECTED MOMENTS SUMMARY =======")
    
    # Count by type
    type_counts = {}
    for moment in moments:
        for score in moment.scores:
            moment_type = score.moment_type.value
            if moment_type not in type_counts:
                type_counts[moment_type] = 0
            type_counts[moment_type] += 1
    
    print("\nMoment Types:")
    for type_name, count in type_counts.items():
        print(f"  - {type_name}: {count}")
    
    # Sort by combined score and print top moments
    top_moments = sorted(moments, key=lambda m: m.combined_score, reverse=True)[:5]
    
    print("\nTop 5 Moments:")
    for i, moment in enumerate(top_moments):
        print(f"\n{i+1}. Time: {moment.start_time:.2f}s - {moment.end_time:.2f}s (Duration: {moment.end_time - moment.start_time:.2f}s)")
        print(f"   Combined Score: {moment.combined_score:.2f}")
        print("   Breakdown:")
        
        for score in sorted(moment.scores, key=lambda s: s.score, reverse=True):
            print(f"     - {score.moment_type.value}: {score.score:.2f} (confidence: {score.confidence:.2f})")
            
            # Print interesting metadata based on moment type
            if score.moment_type == MomentType.SENTIMENT_PEAK:
                if 'sentiment_value' in score.metadata:
                    sentiment = score.metadata['sentiment_value']
                    print(f"       Sentiment: {sentiment:.2f} ({'positive' if sentiment > 0 else 'negative'})")
                if 'sentiment_text' in score.metadata:
                    print(f"       Text: \"{score.metadata['sentiment_text']}\"")
                    
            elif score.moment_type == MomentType.KEYWORD:
                if 'keyword' in score.metadata:
                    print(f"       Keyword: \"{score.metadata['keyword']}\"")
                if 'importance' in score.metadata:
                    print(f"       Importance: {score.metadata['importance']:.2f}")
                    
            elif score.moment_type == MomentType.LAUGHTER:
                if 'intensity' in score.metadata:
                    print(f"       Intensity: {score.metadata['intensity']:.2f}")
                    
        # Print transcript if available
        if moment.transcript:
            print(f"   Transcript: \"{moment.transcript}\"")


def main():
    parser = argparse.ArgumentParser(description="Interesting Moment Detection Demo")
    parser.add_argument("video_path", help="Path to the video file to analyze")
    parser.add_argument("--transcript", "-t", help="Optional path to a transcript file")
    parser.add_argument("--output-dir", "-o", default="output", help="Directory to save outputs")
    parser.add_argument("--extract-clips", "-e", action="store_true", help="Extract highlight clips")
    parser.add_argument("--visualize", "-v", action="store_true", help="Generate visualization")
    parser.add_argument("--clip-count", "-c", type=int, default=3, help="Number of highlight clips to extract")
    parser.add_argument("--min-duration", "-min", type=float, default=3.0, help="Minimum clip duration in seconds")
    parser.add_argument("--max-duration", "-max", type=float, default=15.0, help="Maximum clip duration in seconds")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input
    if not os.path.exists(args.video_path):
        logger.error(f"Video file not found: {args.video_path}")
        return 1
    
    if args.transcript and not os.path.exists(args.transcript):
        logger.error(f"Transcript file not found: {args.transcript}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create analyzer config
    config = MomentAnalyzerConfig(
        temp_dir=str(output_dir / "temp"),
        output_dir=str(output_dir),
        verbose_logging=args.debug,
        min_moment_duration=args.min_duration,
        max_moment_duration=args.max_duration
    )
    
    # Initialize analyzer
    analyzer = MomentAnalyzer(config)
    
    # Start analysis
    logger.info(f"Analyzing video: {args.video_path}")
    start_time = os.times().elapsed
    
    # Analyze the video
    moments = analyzer.analyze_video(args.video_path, args.transcript)
    
    # Calculate duration
    duration = os.times().elapsed - start_time
    logger.info(f"Analysis completed in {duration:.2f} seconds")
    logger.info(f"Detected {len(moments)} interesting moments")
    
    # Print summary
    summarize_moments(moments)
    
    # Save results as JSON
    results_path = output_dir / f"{Path(args.video_path).stem}_moments.json"
    with open(results_path, 'w') as f:
        json.dump(
            {
                "video_path": args.video_path,
                "analysis_duration_seconds": duration,
                "moment_count": len(moments),
                "moments": [moment.to_dict() for moment in moments]
            },
            f,
            indent=2
        )
    logger.info(f"Results saved to {results_path}")
    
    # Generate visualization if requested
    if args.visualize:
        # Get video duration
        import subprocess
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            args.video_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                video_duration = float(result.stdout.strip())
            else:
                logger.warning("Could not determine video duration, using estimate")
                video_duration = max([m.end_time for m in moments]) + 10
        except Exception as e:
            logger.warning(f"Error getting video duration: {str(e)}")
            video_duration = max([m.end_time for m in moments]) + 10
        
        viz_path = output_dir / f"{Path(args.video_path).stem}_visualization.png"
        visualize_moments(moments, video_duration, viz_path)
    
    # Extract clips if requested
    if args.extract_clips:
        highlights = analyzer.extract_highlights(
            args.video_path,
            str(output_dir),
            max_highlights=args.clip_count,
            min_duration=args.min_duration,
            max_duration=args.max_duration
        )
        
        logger.info(f"Extracted {len(highlights)} highlight clips")
        for highlight in highlights:
            logger.info(f"  - {highlight['output_path']} ({highlight['duration']:.2f}s)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 