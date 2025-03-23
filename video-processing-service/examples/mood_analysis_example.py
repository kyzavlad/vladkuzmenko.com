#!/usr/bin/env python3
"""
Content Mood Analysis Example

This script demonstrates the usage of the MoodAnalyzer class to analyze
the emotional mood of video content for music selection.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.music.mood_analyzer import MoodAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def format_mood_scores(mood_scores: Dict[str, float], top_n: int = 5) -> str:
    """Format mood scores for display."""
    sorted_moods = sorted(mood_scores.items(), key=lambda x: x[1], reverse=True)
    top_moods = sorted_moods[:top_n]
    
    max_mood_len = max(len(mood) for mood, _ in top_moods)
    result = []
    
    for mood, score in top_moods:
        # Create a visual bar representation of the score
        bar_length = int(score * 30)
        bar = '█' * bar_length
        result.append(f"{mood.ljust(max_mood_len)}: {score:.2f} {bar}")
    
    return '\n'.join(result)

def format_va_coordinates(valence: float, arousal: float) -> str:
    """Format valence-arousal coordinates for display."""
    # Create a simple 2D grid representation
    grid_size = 11
    grid = [[' ' for _ in range(grid_size)] for _ in range(grid_size)]
    
    # Add axes and labels
    for i in range(grid_size):
        grid[i][grid_size // 2] = '│'
        grid[grid_size // 2][i] = '─'
    
    grid[grid_size // 2][grid_size // 2] = '┼'
    
    # Add point
    v_pos = int((valence + 1) * (grid_size - 1) / 2)
    a_pos = grid_size - 1 - int(arousal * (grid_size - 1))
    
    # Ensure within bounds
    v_pos = max(0, min(grid_size - 1, v_pos))
    a_pos = max(0, min(grid_size - 1, a_pos))
    
    grid[a_pos][v_pos] = 'X'
    
    # Add axes labels
    result = ['  Valence-Arousal Space:']
    result.append('  ' + ' ' * (grid_size // 2 - 4) + 'Negative' + ' ' * (grid_size // 2 - 3) + 'Positive')
    
    for i, row in enumerate(grid):
        prefix = 'High  ' if i == 0 else '      ' if i != grid_size - 1 else 'Low   '
        if i == grid_size // 2:
            prefix = 'Arousal'
        result.append(prefix + ''.join(row))
    
    return '\n'.join(result)

def main():
    """Main function to demonstrate content mood analysis."""
    parser = argparse.ArgumentParser(description="Content mood analysis example")
    parser.add_argument('video', type=str, help='Path to input video file')
    parser.add_argument('--transcript', type=str, help='Path to transcript JSON file (optional)')
    parser.add_argument('--output', type=str, help='Path to output JSON file (optional)')
    parser.add_argument('--segment-duration', type=int, default=5, help='Duration of segments for timeline analysis (seconds)')
    parser.add_argument('--no-audio', action='store_true', help='Skip audio analysis')
    parser.add_argument('--no-visual', action='store_true', help='Skip visual analysis')
    parser.add_argument('--no-transcript', action='store_true', help='Skip transcript analysis')
    parser.add_argument('--ffmpeg', type=str, default='ffmpeg', help='Path to ffmpeg')
    parser.add_argument('--ffprobe', type=str, default='ffprobe', help='Path to ffprobe')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.video):
        logger.error(f"Video file not found: {args.video}")
        return 1
    
    # Load transcript if provided
    transcript = None
    if args.transcript:
        if not os.path.exists(args.transcript):
            logger.error(f"Transcript file not found: {args.transcript}")
            return 1
        
        try:
            with open(args.transcript, 'r') as f:
                transcript = json.load(f)
            logger.info(f"Loaded transcript with {len(transcript)} segments")
        except Exception as e:
            logger.error(f"Error loading transcript: {str(e)}")
            return 1
    
    # Configure mood analyzer
    config = {
        'ffmpeg_path': args.ffmpeg,
        'ffprobe_path': args.ffprobe,
        'segment_duration': args.segment_duration
    }
    
    # Create mood analyzer
    analyzer = MoodAnalyzer(config)
    
    logger.info(f"Analyzing mood of video: {args.video}")
    logger.info(f"Analysis components: " + 
               f"Audio={'Yes' if not args.no_audio else 'No'}, " +
               f"Visual={'Yes' if not args.no_visual else 'No'}, " +
               f"Transcript={'Yes' if not args.no_transcript and transcript else 'No'}")
    
    # Analyze mood
    result = analyzer.analyze_mood(
        video_path=args.video,
        transcript=transcript,
        include_audio_analysis=not args.no_audio,
        include_visual_analysis=not args.no_visual,
        include_transcript_analysis=not args.no_transcript and transcript is not None
    )
    
    # Display results
    print("\n" + "="*60)
    print(f"MOOD ANALYSIS RESULTS FOR: {os.path.basename(args.video)}")
    print("="*60)
    
    print(f"\nPRIMARY MOOD: {result['primary_mood'].upper()}")
    print(f"Valence: {result['valence']:.2f}, Arousal: {result['arousal']:.2f}")
    
    print("\n" + format_va_coordinates(result['valence'], result['arousal']))
    
    print("\nTOP MOOD SCORES:")
    print(format_mood_scores(result['mood_scores']))
    
    print("\nRECOMMENDED MUSIC MOODS:")
    for mood in result['recommended_music_moods']:
        print(f"- {mood}")
    
    # Show timeline if available
    if 'timeline' in result and result['timeline']:
        print("\nMOOD TIMELINE:")
        for segment in result['timeline'][:5]:  # Show first 5 segments
            print(f"- {segment['start_time']:.1f}s - {segment['end_time']:.1f}s: {segment['mood']}")
        
        if len(result['timeline']) > 5:
            print(f"  ... and {len(result['timeline']) - 5} more segments")
    
    # Show component contributions if available
    if 'component_weights' in result:
        print("\nANALYSIS COMPONENT WEIGHTS:")
        for component, weight in result['component_weights'].items():
            print(f"- {component}: {weight:.2f}")
    
    # Save to file if output path provided
    if args.output:
        output_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(output_dir, exist_ok=True)
        
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results saved to: {args.output}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 