#!/usr/bin/env python3
"""
Emotional Arc Mapping Example

This script demonstrates the usage of the EmotionalArcMapper class to analyze
the emotional arc of video content for synchronized music selection.
"""

import os
import sys
import argparse
import logging
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.music.emotional_arc_mapper import EmotionalArcMapper
from app.services.music.mood_analyzer import MoodAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def visualize_arc(
    arc_data: List[Dict[str, Any]], 
    key_moments: Optional[List[Dict[str, Any]]] = None,
    music_cues: Optional[List[Dict[str, Any]]] = None,
    output_path: Optional[str] = None
):
    """
    Visualize the emotional arc with key moments and music cues.
    
    Args:
        arc_data: List of emotional arc segments
        key_moments: List of key emotional moments (optional)
        music_cues: List of music cue points (optional)
        output_path: Path to save the visualization (optional)
    """
    if not arc_data:
        logger.error("No emotional arc data to visualize")
        return
    
    # Extract data for plotting
    times = [(segment["start_time"] + segment["end_time"]) / 2 for segment in arc_data]
    valence = [segment["valence"] for segment in arc_data]
    arousal = [segment["arousal"] for segment in arc_data]
    intensity = [segment["emotional_intensity"] for segment in arc_data]
    
    # Create figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot valence and arousal
    ax1.plot(times, valence, 'b-', label='Valence', linewidth=2)
    ax1.plot(times, arousal, 'r-', label='Arousal', linewidth=2)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_ylabel('Value')
    ax1.set_title('Emotional Arc: Valence and Arousal')
    ax1.legend()
    ax1.set_ylim(-1.1, 1.1)
    ax1.grid(True, alpha=0.3)
    
    # Plot emotional intensity
    ax2.plot(times, intensity, 'g-', label='Emotional Intensity', linewidth=2)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Intensity')
    ax2.set_title('Emotional Intensity')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    
    # Add key moments if available
    if key_moments:
        for moment in key_moments:
            moment_time = moment["time"]
            if moment["type"] == "peak":
                marker = '^'
                color = 'orange'
            elif moment["type"] == "valley":
                marker = 'v'
                color = 'purple'
            else:
                marker = 'o'
                color = 'black'
            
            # Add markers to both plots
            ax1.plot(moment_time, moment["valence"], marker=marker, markersize=8, color=color)
            ax1.plot(moment_time, moment["arousal"], marker=marker, markersize=8, color=color)
            ax2.plot(moment_time, moment["emotional_intensity"], marker=marker, markersize=8, color=color)
            
            # Add vertical line
            ax1.axvline(x=moment_time, color=color, linestyle='--', alpha=0.5)
            ax2.axvline(x=moment_time, color=color, linestyle='--', alpha=0.5)
            
            # Add text annotation
            ax1.annotate(
                moment["moment_mood"],
                xy=(moment_time, 0.9),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                rotation=90,
                fontsize=8,
                alpha=0.7
            )
    
    # Add music cues if available
    if music_cues:
        for cue in music_cues:
            cue_time = cue["time"]
            if cue["cue_type"] == "intro":
                color = 'green'
                label = 'Intro'
            elif cue["cue_type"] == "outro":
                color = 'red'
                label = 'Outro'
            else:
                color = 'magenta'
                label = cue["cue_type"].capitalize()
            
            # Add vertical line
            ax2.axvline(x=cue_time, color=color, linestyle='-', alpha=0.5)
            
            # Add text annotation
            ax2.annotate(
                label,
                xy=(cue_time, 0.1),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                rotation=90,
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
            )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the figure
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    """Main function to demonstrate emotional arc mapping."""
    parser = argparse.ArgumentParser(description="Emotional arc mapping example")
    parser.add_argument('video', type=str, help='Path to input video file')
    parser.add_argument('--transcript', type=str, help='Path to transcript JSON file (optional)')
    parser.add_argument('--output', type=str, help='Path to output JSON file (optional)')
    parser.add_argument('--segment-duration', type=int, default=5, help='Duration of segments for timeline analysis (seconds)')
    parser.add_argument('--no-key-moments', action='store_true', help='Skip key moment detection')
    parser.add_argument('--no-smoothing', action='store_true', help='Skip arc smoothing')
    parser.add_argument('--visualize', action='store_true', help='Create visualization')
    parser.add_argument('--vis-output', type=str, help='Path to save visualization (requires --visualize)')
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
    
    # Configure emotional arc mapper
    config = {
        'ffmpeg_path': args.ffmpeg,
        'ffprobe_path': args.ffprobe,
        'segment_duration': args.segment_duration
    }
    
    # Create mapper
    mapper = EmotionalArcMapper(config)
    
    logger.info(f"Mapping emotional arc of video: {args.video}")
    
    # Map emotional arc
    result = mapper.map_emotional_arc(
        video_path=args.video,
        transcript=transcript,
        segment_duration=args.segment_duration,
        detect_key_moments=not args.no_key_moments,
        smooth_arc=not args.no_smoothing
    )
    
    # Display results
    print("\n" + "="*70)
    print(f"EMOTIONAL ARC MAPPING FOR: {os.path.basename(args.video)}")
    print("="*70)
    
    # Display overall pattern
    pattern = result.get("arc_pattern", "unknown")
    confidence = result.get("arc_confidence", 0)
    description = result.get("pattern_description", "")
    
    print(f"\nEMOTIONAL ARC PATTERN: {pattern.upper()} (Confidence: {confidence:.2f})")
    print(f"Description: {description}")
    
    # Display emotional dynamics
    if "emotional_dynamics" in result:
        dynamics = result["emotional_dynamics"]
        print("\nEMOTIONAL DYNAMICS:")
        if "emotional_range" in dynamics:
            ranges = dynamics["emotional_range"]
            print(f"- Emotional Range: Valence {ranges['valence_range'][0]:.2f} to {ranges['valence_range'][1]:.2f}, "
                  f"Arousal {ranges['arousal_range'][0]:.2f} to {ranges['arousal_range'][1]:.2f}")
        
        print(f"- Emotional Variability: {dynamics.get('emotional_variability', 0):.2f}")
        print(f"- Mood Diversity: {dynamics.get('mood_diversity', 0)} distinct moods")
        print(f"- Emotional Complexity: {dynamics.get('emotional_complexity', 'unknown')}")
    
    # Show key moments if available
    if "key_moments" in result and result["key_moments"]:
        moments = result["key_moments"]
        print(f"\nKEY EMOTIONAL MOMENTS: ({len(moments)})")
        for i, moment in enumerate(moments, 1):
            moment_type = moment["type"]
            mood = moment["moment_mood"]
            time = moment["time"]
            transition = moment["transition_type"]
            
            print(f"- [{i}] {time:.1f}s: {transition.capitalize()} {moment_type} ({mood})")
    
    # Show music cues if available
    if "music_cues" in result and result["music_cues"]:
        cues = result["music_cues"]
        print(f"\nMUSIC CUE POINTS: ({len(cues)})")
        for i, cue in enumerate(cues, 1):
            cue_type = cue["cue_type"]
            time = cue["time"]
            mood = cue["mood"]
            
            print(f"- [{i}] {time:.1f}s: {cue_type.capitalize()} cue - {mood}")
    
    # Create visualization if requested
    if args.visualize:
        try:
            import matplotlib
            matplotlib.use('Agg' if args.vis_output else 'TkAgg')
            
            arc_data = result.get("emotional_arc", [])
            key_moments = result.get("key_moments", []) if not args.no_key_moments else None
            music_cues = result.get("music_cues", [])
            
            visualize_arc(
                arc_data=arc_data,
                key_moments=key_moments,
                music_cues=music_cues,
                output_path=args.vis_output
            )
        except ImportError:
            logger.warning("Matplotlib not available. Visualization skipped.")
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
    
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