#!/usr/bin/env python3
"""
BPM Detection Example

This script demonstrates the usage of the BPMDetector class to analyze
and match audio files based on tempo (BPM).
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.music.bpm_detector import BPMDetector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to demonstrate BPM detection and matching"""
    parser = argparse.ArgumentParser(description="BPM detection and matching example")
    parser.add_argument('--audio', type=str, help='Path to audio file or directory')
    parser.add_argument('--target-bpm', type=float, help='Target BPM to match')
    parser.add_argument('--tolerance', type=float, default=5.0, help='BPM tolerance range (+/-)')
    parser.add_argument('--match-style', type=str, default='exact', 
                        choices=['exact', 'double', 'harmonic'], 
                        help='Matching style for BPM')
    parser.add_argument('--content-type', type=str, help='Content type for BPM suggestion')
    parser.add_argument('--output', type=str, help='Path to output JSON file (optional)')
    parser.add_argument('--ffmpeg', type=str, default='ffmpeg', help='Path to ffmpeg')
    parser.add_argument('--ffprobe', type=str, default='ffprobe', help='Path to ffprobe')
    
    args = parser.parse_args()
    
    if not args.audio and not args.content_type:
        parser.print_help()
        return
    
    # Configure the BPM detector
    config = {
        'ffmpeg_path': args.ffmpeg,
        'ffprobe_path': args.ffprobe,
    }
    
    detector = BPMDetector(config)
    
    results = {}
    
    # Suggest BPM for content type
    if args.content_type:
        suggestion = detector.suggest_bpm_for_content(args.content_type)
        print_bpm_suggestion(suggestion)
        results['suggestion'] = suggestion
    
    # Process audio file or directory
    if args.audio:
        audio_path = Path(args.audio)
        
        if audio_path.is_file():
            # Single file BPM detection
            detection_result = process_file(detector, audio_path)
            results['detection'] = detection_result
        
        elif audio_path.is_dir():
            # Directory processing - BPM detection and matching
            audio_files = find_audio_files(audio_path)
            
            if not audio_files:
                logger.error(f"No audio files found in {audio_path}")
                return
            
            logger.info(f"Found {len(audio_files)} audio files")
            
            # Detect BPM for all files
            detection_results = []
            for file_path in audio_files:
                detection = process_file(detector, file_path)
                if detection:
                    detection_results.append(detection)
            
            results['detection_results'] = detection_results
            
            # BPM matching if target specified
            if args.target_bpm:
                match_result = detector.find_matching_tracks(
                    target_bpm=args.target_bpm,
                    audio_files=[str(f) for f in audio_files],
                    tolerance=args.tolerance,
                    match_style=args.match_style
                )
                
                print_match_results(match_result)
                results['matching'] = match_result
    
    # Save results if output file specified
    if args.output and results:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")

def process_file(detector: BPMDetector, file_path: Path) -> Dict[str, Any]:
    """Process a single audio file for BPM detection"""
    logger.info(f"Analyzing {file_path}")
    
    try:
        detection = detector.detect_bpm(str(file_path))
        
        if detection['status'] == 'success':
            print(f"\n🎵 BPM analysis for {file_path.name}:")
            print(f"  BPM: {detection['bpm']:.1f}")
            print(f"  Category: {detection['category']}")
            print(f"  Confidence: {detection['confidence']:.2f}" if detection.get('confidence') else "  Confidence: N/A")
            
            return detection
        else:
            logger.error(f"BPM detection failed: {detection.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return None

def find_audio_files(directory: Path) -> List[Path]:
    """Find audio files in a directory"""
    audio_extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac']
    
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(directory.glob(f'*{ext}'))
    
    return audio_files

def print_match_results(match_result: Dict[str, Any]):
    """Print BPM matching results"""
    if match_result['status'] != 'success':
        print(f"\n❌ Matching failed: {match_result.get('error', 'Unknown error')}")
        return
    
    print(f"\n🎯 BPM matching results for target {match_result['target_bpm']} BPM:")
    print(f"  Tolerance: ±{match_result['tolerance']} BPM")
    print(f"  Match style: {match_result['match_style']}")
    
    if not match_result['matches']:
        print("  No matching tracks found")
        return
    
    print("\nMatches (sorted by match score):")
    for i, match in enumerate(match_result['matches'], 1):
        filename = os.path.basename(match['file_path'])
        print(f"  {i}. {filename}")
        print(f"     BPM: {match['bpm']:.1f} ({match['category']})")
        print(f"     Match score: {match['match_score']:.2f}")
        print(f"     Difference: {match['bpm_difference']:.1f} BPM")
        print()
    
    if match_result['errors']:
        print(f"\n{len(match_result['errors'])} files had errors during processing")

def print_bpm_suggestion(suggestion: Dict[str, Any]):
    """Print BPM suggestion for content type"""
    if suggestion['status'] != 'success' and suggestion['status'] != 'warning':
        print(f"\n❌ Suggestion failed: {suggestion.get('error', 'Unknown error')}")
        return
    
    print(f"\n🎬 BPM suggestion for {suggestion['content_type']} content:")
    if suggestion['status'] == 'warning':
        print(f"  ⚠️ {suggestion['message']}")
    
    print(f"  Suggested BPM: {suggestion['suggested_bpm']:.1f}")
    print(f"  BPM range: {suggestion['bpm_range'][0]}-{suggestion['bpm_range'][1]}")
    print(f"  Category: {suggestion['category']}")
    print()

if __name__ == "__main__":
    main() 