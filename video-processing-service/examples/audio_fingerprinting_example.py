#!/usr/bin/env python3
"""
Audio Fingerprinting Example

This script demonstrates the usage of the AudioFingerprinter class for audio identification,
fingerprint comparison, and management of the fingerprint database.
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

from app.services.music.audio_fingerprinter import AudioFingerprinter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_fingerprinter(config: Dict[str, Any]) -> AudioFingerprinter:
    """Initialize the audio fingerprinter with configuration."""
    logger.info("Initializing audio fingerprinter...")
    return AudioFingerprinter(config)

def generate_fingerprint_example(fingerprinter: AudioFingerprinter, audio_path: str) -> Dict[str, Any]:
    """
    Demonstrate generating an audio fingerprint.
    
    Args:
        fingerprinter: AudioFingerprinter instance
        audio_path: Path to the audio file
        
    Returns:
        Fingerprint result
    """
    logger.info(f"Generating fingerprint for: {audio_path}")
    result = fingerprinter.generate_fingerprint(audio_path)
    
    if result["status"] == "success":
        logger.info("Fingerprint generated successfully")
        
        # Print fingerprint hash
        fingerprint = result.get("fingerprint", {})
        if "hash" in fingerprint:
            logger.info(f"Fingerprint hash: {fingerprint['hash']}")
        
        # Print vector information
        if "vector" in fingerprint:
            vector_length = len(fingerprint["vector"])
            logger.info(f"Vector length: {vector_length}")
            if vector_length > 0:
                logger.info(f"Vector sample: {fingerprint['vector'][:3]}...")
    else:
        logger.error(f"Failed to generate fingerprint: {result.get('error', 'Unknown error')}")
    
    return result

def add_to_database_example(fingerprinter: AudioFingerprinter, audio_path: str, track_id: str, title: str, artist: str) -> Dict[str, Any]:
    """
    Demonstrate adding a fingerprint to the database.
    
    Args:
        fingerprinter: AudioFingerprinter instance
        audio_path: Path to the audio file
        track_id: ID of the track
        title: Title of the track
        artist: Artist of the track
        
    Returns:
        Result of the operation
    """
    logger.info(f"Adding fingerprint to database for: {audio_path}")
    
    metadata = {
        "genre": "example",
        "duration": 180,
        "source": "example script"
    }
    
    result = fingerprinter.add_to_database(
        audio_path=audio_path,
        track_id=track_id,
        title=title,
        artist=artist,
        metadata=metadata
    )
    
    if result["status"] == "success":
        logger.info(f"Fingerprint added to database with ID: {track_id}")
    else:
        logger.error(f"Failed to add fingerprint to database: {result.get('error', 'Unknown error')}")
    
    return result

def identify_audio_example(fingerprinter: AudioFingerprinter, audio_path: str) -> Dict[str, Any]:
    """
    Demonstrate identifying an audio file.
    
    Args:
        fingerprinter: AudioFingerprinter instance
        audio_path: Path to the audio file
        
    Returns:
        Identification result
    """
    logger.info(f"Identifying audio: {audio_path}")
    result = fingerprinter.identify_audio(audio_path)
    
    if result["status"] == "success":
        matches = result.get("matches", [])
        match_count = len(matches)
        
        if match_count > 0:
            logger.info(f"Found {match_count} matches")
            for i, match in enumerate(matches, 1):
                logger.info(f"Match {i}:")
                logger.info(f"  Track ID: {match.get('track_id', 'Unknown')}")
                logger.info(f"  Title: {match.get('title', 'Unknown')}")
                logger.info(f"  Artist: {match.get('artist', 'Unknown')}")
                logger.info(f"  Similarity: {match.get('similarity', 0):.2f}")
        else:
            logger.info("No matches found")
    else:
        logger.error(f"Failed to identify audio: {result.get('error', 'Unknown error')}")
    
    return result

def compare_fingerprints_example(fingerprinter: AudioFingerprinter, audio_path1: str, audio_path2: str) -> Dict[str, Any]:
    """
    Demonstrate comparing two audio files.
    
    Args:
        fingerprinter: AudioFingerprinter instance
        audio_path1: Path to the first audio file
        audio_path2: Path to the second audio file
        
    Returns:
        Comparison result
    """
    logger.info(f"Comparing audio files:")
    logger.info(f"  File 1: {audio_path1}")
    logger.info(f"  File 2: {audio_path2}")
    
    # Generate fingerprints for both files
    fingerprint1 = fingerprinter.generate_fingerprint(audio_path1)
    fingerprint2 = fingerprinter.generate_fingerprint(audio_path2)
    
    # Check if fingerprints were successfully generated
    if fingerprint1["status"] != "success":
        logger.error(f"Failed to generate fingerprint for {audio_path1}")
        return fingerprint1
    
    if fingerprint2["status"] != "success":
        logger.error(f"Failed to generate fingerprint for {audio_path2}")
        return fingerprint2
    
    # Compare fingerprints
    comparison = fingerprinter.compare_fingerprints(
        fingerprint1["fingerprint"],
        fingerprint2["fingerprint"]
    )
    
    if comparison["status"] == "success":
        logger.info(f"Comparison results:")
        logger.info(f"  Similarity: {comparison.get('similarity', 0):.4f}")
        logger.info(f"  Distance: {comparison.get('distance', 0):.4f}")
        logger.info(f"  Match: {comparison.get('is_match', False)}")
    else:
        logger.error(f"Failed to compare fingerprints: {comparison.get('error', 'Unknown error')}")
    
    return comparison

def view_database_example(fingerprinter: AudioFingerprinter) -> Dict[str, Any]:
    """
    Demonstrate viewing the fingerprint database.
    
    Args:
        fingerprinter: AudioFingerprinter instance
        
    Returns:
        Database information
    """
    logger.info("Loading fingerprint database")
    database = fingerprinter._load_fingerprint_database()
    
    logger.info(f"Database contains {len(database)} fingerprints")
    logger.info(f"Database path: {fingerprinter.fingerprint_db_path}")
    
    # Display fingerprint entries
    for i, entry in enumerate(database, 1):
        logger.info(f"Entry {i}:")
        logger.info(f"  Track ID: {entry.get('track_id', 'Unknown')}")
        logger.info(f"  Title: {entry.get('title', 'Unknown')}")
        logger.info(f"  Artist: {entry.get('artist', 'Unknown')}")
        logger.info(f"  Added: {entry.get('timestamp', 'Unknown')}")
    
    return {
        "status": "success",
        "count": len(database),
        "entries": database
    }

def remove_from_database_example(fingerprinter: AudioFingerprinter, track_id: str) -> Dict[str, Any]:
    """
    Demonstrate removing a fingerprint from the database.
    
    Args:
        fingerprinter: AudioFingerprinter instance
        track_id: ID of the track to remove
        
    Returns:
        Result of the operation
    """
    logger.info(f"Removing fingerprint with ID: {track_id}")
    result = fingerprinter.remove_from_database(track_id)
    
    if result["status"] == "success":
        logger.info("Fingerprint removed successfully")
    else:
        logger.error(f"Failed to remove fingerprint: {result.get('error', 'Unknown error')}")
    
    return result

def main():
    """Main function to demonstrate audio fingerprinting."""
    parser = argparse.ArgumentParser(description="Audio fingerprinting example")
    
    # General arguments
    parser.add_argument('--db-path', type=str, default='./fingerprint_db',
                        help='Path to fingerprint database directory')
    parser.add_argument('--ffmpeg', type=str, default='ffmpeg',
                        help='Path to ffmpeg executable')
    parser.add_argument('--ffprobe', type=str, default='ffprobe',
                        help='Path to ffprobe executable')
    
    # Mode selection
    parser.add_argument('--generate', action='store_true',
                        help='Generate a fingerprint')
    parser.add_argument('--add', action='store_true',
                        help='Add a fingerprint to the database')
    parser.add_argument('--identify', action='store_true',
                        help='Identify an audio file')
    parser.add_argument('--compare', action='store_true',
                        help='Compare two audio files')
    parser.add_argument('--view-db', action='store_true',
                        help='View the fingerprint database')
    parser.add_argument('--remove', action='store_true',
                        help='Remove a fingerprint from the database')
    
    # File arguments
    parser.add_argument('--audio-file', type=str,
                        help='Path to the audio file')
    parser.add_argument('--audio-file2', type=str,
                        help='Path to the second audio file (for comparison)')
    
    # Track information
    parser.add_argument('--track-id', type=str,
                        help='ID of the track')
    parser.add_argument('--title', type=str, default='Example Track',
                        help='Title of the track')
    parser.add_argument('--artist', type=str, default='Example Artist',
                        help='Artist of the track')
    
    args = parser.parse_args()
    
    # Configure the audio fingerprinter
    config = {
        'fingerprint_db_path': args.db_path,
        'ffmpeg_path': args.ffmpeg,
        'ffprobe_path': args.ffprobe,
        'similarity_threshold': 0.85,  # Default similarity threshold
        'distance_threshold': 0.5     # Default distance threshold
    }
    
    # Initialize the audio fingerprinter
    fingerprinter = init_fingerprinter(config)
    
    # Execute the requested operation
    if args.generate:
        if not args.audio_file:
            parser.error("--generate requires --audio-file")
        generate_fingerprint_example(fingerprinter, args.audio_file)
    
    elif args.add:
        if not args.audio_file or not args.track_id:
            parser.error("--add requires --audio-file and --track-id")
        add_to_database_example(fingerprinter, args.audio_file, args.track_id, args.title, args.artist)
    
    elif args.identify:
        if not args.audio_file:
            parser.error("--identify requires --audio-file")
        identify_audio_example(fingerprinter, args.audio_file)
    
    elif args.compare:
        if not args.audio_file or not args.audio_file2:
            parser.error("--compare requires --audio-file and --audio-file2")
        compare_fingerprints_example(fingerprinter, args.audio_file, args.audio_file2)
    
    elif args.view_db:
        view_database_example(fingerprinter)
    
    elif args.remove:
        if not args.track_id:
            parser.error("--remove requires --track-id")
        remove_from_database_example(fingerprinter, args.track_id)
    
    else:
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 