#!/usr/bin/env python3
"""
Genre Classification Example

This script demonstrates the usage of the GenreClassifier class to analyze
audio files and recommend music genres for video content.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.music.genre_classifier import GenreClassifier

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to demonstrate genre classification"""
    parser = argparse.ArgumentParser(description="Music genre classification example")
    parser.add_argument('--audio', type=str, help='Path to audio file or directory')
    parser.add_argument('--output', type=str, help='Path to output JSON file (optional)')
    parser.add_argument('--video-mood', type=str, default='energetic',
                        help='Video mood for genre recommendations (e.g., happy, sad, energetic)')
    parser.add_argument('--video-genre', type=str, default='documentary',
                        help='Video genre for recommendations (e.g., documentary, tutorial)')
    parser.add_argument('--ffmpeg', type=str, default='ffmpeg', help='Path to ffmpeg')
    parser.add_argument('--ffprobe', type=str, default='ffprobe', help='Path to ffprobe')
    
    args = parser.parse_args()
    
    if not args.audio:
        parser.print_help()
        return
    
    # Configure the genre classifier
    config = {
        'ffmpeg_path': args.ffmpeg,
        'ffprobe_path': args.ffprobe,
    }
    
    classifier = GenreClassifier(config)
    
    # Recommend genres based on video content
    logger.info(f"Recommending music genres for {args.video_mood} {args.video_genre} video")
    recommendations = classifier.recommend_genres_for_video(
        args.video_mood, args.video_genre
    )
    
    print("\n🎵 Recommended genres for your video:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['genre']} (score: {rec['score']:.2f})")
    
    # Process audio file or directory
    results = []
    audio_path = Path(args.audio)
    
    if audio_path.is_file():
        result = process_file(classifier, audio_path)
        if result:
            results.append(result)
    elif audio_path.is_dir():
        for file_path in audio_path.glob('*.mp3'):
            result = process_file(classifier, file_path)
            if result:
                results.append(result)
        
        for file_path in audio_path.glob('*.wav'):
            result = process_file(classifier, file_path)
            if result:
                results.append(result)
    
    # Save results if output file specified
    if args.output and results:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")

def process_file(classifier, file_path):
    """Process a single audio file"""
    logger.info(f"Analyzing {file_path}")
    
    try:
        classification = classifier.classify_genre(str(file_path))
        
        if classification['status'] == 'success':
            print(f"\n🎵 Genre analysis for {file_path.name}:")
            print(f"Primary genre: {classification['primary_genre']}")
            print("Top genres:")
            for i, genre in enumerate(classification['top_genres'], 1):
                print(f"  {i}. {genre['genre']} ({genre['probability']:.2f})")
            
            if 'feature_summary' in classification:
                print("\nAudio characteristics:")
                for feature, value in classification['feature_summary'].items():
                    print(f"  {feature}: {value}")
            
            return classification
        else:
            logger.error(f"Classification failed: {classification.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return None

if __name__ == "__main__":
    main() 