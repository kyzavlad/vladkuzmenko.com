#!/usr/bin/env python3
"""
Music Library Example

This script demonstrates the usage of the MusicLibrary class for managing
music tracks, collections, and searching based on various criteria.
"""

import os
import sys
import argparse
import logging
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.music.music_library import MusicLibrary

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_music_library(config: Dict[str, Any]) -> MusicLibrary:
    """Initialize the music library with configuration."""
    logger.info("Initializing music library...")
    return MusicLibrary(config)

def add_track_example(library: MusicLibrary, args: argparse.Namespace) -> None:
    """Demonstrate adding a track to the library."""
    if not args.audio_file:
        logger.warning("No audio file provided. Skipping add track example.")
        return
    
    logger.info(f"Adding track '{os.path.basename(args.audio_file)}' to the library...")
    
    result = library.add_track(
        file_path=args.audio_file,
        title=args.title or "Example Track",
        artist=args.artist or "Example Artist",
        mood=args.mood,
        genre=args.genre,
        bpm=args.bpm,
        tags=args.tags.split(",") if args.tags else ["example", "demo"],
        description=args.description or "Example track added for demonstration",
        copyright_free=args.copyright_free
    )
    
    if result["status"] == "success":
        logger.info(f"Track added successfully with ID: {result['track_id']}")
        return result["track_id"]
    else:
        logger.error(f"Failed to add track: {result['error']}")
        return None

def create_collection_example(library: MusicLibrary, args: argparse.Namespace) -> Optional[str]:
    """Demonstrate creating a collection."""
    logger.info("Creating a music collection...")
    
    result = library.create_collection(
        name=args.collection_name or "Example Collection",
        description=args.collection_description or "A collection of example tracks",
        tags=args.collection_tags.split(",") if args.collection_tags else ["example", "demo"]
    )
    
    if result["status"] == "success":
        collection_id = result["collection_id"]
        logger.info(f"Collection created successfully with ID: {collection_id}")
        return collection_id
    else:
        logger.error(f"Failed to create collection: {result['error']}")
        return None

def add_tracks_to_collection_example(library: MusicLibrary, collection_id: str, track_ids: List[str]) -> None:
    """Demonstrate adding tracks to a collection."""
    if not collection_id or not track_ids:
        logger.warning("Missing collection ID or track IDs. Skipping add to collection example.")
        return
    
    logger.info(f"Adding {len(track_ids)} tracks to collection {collection_id}...")
    
    result = library.add_tracks_to_collection(
        collection_id=collection_id,
        track_ids=track_ids
    )
    
    if result["status"] == "success":
        logger.info(f"Added {len(result.get('added_tracks', []))} tracks to collection")
        if result.get('existing_tracks', []):
            logger.info(f"{len(result['existing_tracks'])} tracks were already in the collection")
    else:
        logger.error(f"Failed to add tracks to collection: {result['error']}")

def search_tracks_example(library: MusicLibrary, args: argparse.Namespace) -> None:
    """Demonstrate searching for tracks in the library."""
    logger.info("Searching for tracks based on criteria...")
    
    search_params = {}
    if args.search_mood:
        search_params["mood"] = args.search_mood
    if args.search_genre:
        search_params["genre"] = args.search_genre
    if args.search_bpm is not None:
        search_params["tempo"] = args.search_bpm
    if args.search_keywords:
        search_params["keywords"] = args.search_keywords.split(",")
    if args.search_duration is not None:
        search_params["duration"] = args.search_duration
    if args.search_copyright_free:
        search_params["copyright_free_only"] = True
    if args.search_collection_id:
        search_params["collection_id"] = args.search_collection_id
    
    # Set max results
    search_params["max_results"] = args.max_results
    
    # Search for tracks
    result = library.search_tracks(**search_params)
    
    if result["status"] == "success":
        # Display search results
        print("\n" + "="*60)
        print(f"SEARCH RESULTS ({len(result['tracks'])} of {result.get('total_matches', 0)} matches)")
        print("="*60)
        
        for i, track in enumerate(result["tracks"], 1):
            # Calculate relevance percentage for display
            relevance = track.get("relevance_score", 0) * 100
            
            # Display track information
            print(f"\n{i}. {track.get('title', 'Untitled')} - {track.get('artist', 'Unknown')}")
            print(f"   ID: {track.get('id', 'N/A')}")
            if "mood" in track:
                print(f"   Mood: {track.get('mood')}")
            if "genre" in track:
                print(f"   Genre: {track.get('genre')}")
            if "bpm" in track:
                print(f"   BPM: {track.get('bpm')}")
            if "duration" in track:
                mins, secs = divmod(int(track.get('duration', 0)), 60)
                print(f"   Duration: {mins}:{secs:02d}")
            if "tags" in track and track["tags"]:
                print(f"   Tags: {', '.join(track.get('tags', []))}")
            if "copyright_free" in track:
                copyright_status = "Yes" if track.get('copyright_free') else "No"
                print(f"   Copyright Free: {copyright_status}")
            print(f"   Relevance: {relevance:.1f}%")
        
        # Additional notes
        if "note" in result:
            print(f"\nNote: {result['note']}")
    else:
        logger.error(f"Search failed: {result['error']}")

def display_collections_example(library: MusicLibrary) -> None:
    """Demonstrate displaying all collections."""
    logger.info("Displaying all collections...")
    
    collections = library.get_all_collections()
    
    print("\n" + "="*60)
    print(f"MUSIC COLLECTIONS ({len(collections)})")
    print("="*60)
    
    for i, collection in enumerate(collections, 1):
        # Display collection information
        print(f"\n{i}. {collection.get('name', 'Untitled Collection')}")
        print(f"   ID: {collection.get('id', 'N/A')}")
        if "description" in collection:
            print(f"   Description: {collection.get('description')}")
        if "tags" in collection and collection["tags"]:
            print(f"   Tags: {', '.join(collection.get('tags', []))}")
        
        # Display track count
        track_count = len(collection.get("track_ids", []))
        print(f"   Tracks: {track_count}")
        
        # Display created/updated dates
        if "created_at" in collection:
            print(f"   Created: {collection['created_at']}")
        if "updated_at" in collection:
            print(f"   Last Updated: {collection['updated_at']}")

def display_collection_tracks_example(library: MusicLibrary, collection_id: str) -> None:
    """Demonstrate displaying tracks in a collection."""
    if not collection_id:
        logger.warning("No collection ID provided. Skipping display collection tracks example.")
        return
    
    logger.info(f"Displaying tracks in collection {collection_id}...")
    
    collection = library.get_collection(collection_id)
    if not collection:
        logger.error(f"Collection not found: {collection_id}")
        return
    
    tracks = library.get_collection_tracks(collection_id)
    
    print("\n" + "="*60)
    print(f"TRACKS IN COLLECTION: {collection.get('name', 'Untitled Collection')}")
    print("="*60)
    
    for i, track in enumerate(tracks, 1):
        # Display track information
        print(f"\n{i}. {track.get('title', 'Untitled')} - {track.get('artist', 'Unknown')}")
        print(f"   ID: {track.get('id', 'N/A')}")
        if "mood" in track:
            print(f"   Mood: {track.get('mood')}")
        if "genre" in track:
            print(f"   Genre: {track.get('genre')}")
        if "bpm" in track:
            print(f"   BPM: {track.get('bpm')}")
        if "duration" in track:
            mins, secs = divmod(int(track.get('duration', 0)), 60)
            print(f"   Duration: {mins}:{secs:02d}")

def main():
    """Main function to demonstrate music library functionality."""
    parser = argparse.ArgumentParser(description="Music library example")
    
    # General arguments
    parser.add_argument('--library-path', type=str, default='./music_library',
                        help='Path to music library directory')
    parser.add_argument('--ffmpeg', type=str, default='ffmpeg',
                        help='Path to ffmpeg executable')
    parser.add_argument('--ffprobe', type=str, default='ffprobe',
                        help='Path to ffprobe executable')
    
    # Add track arguments
    parser.add_argument('--audio-file', type=str, help='Path to audio file to add')
    parser.add_argument('--title', type=str, help='Title of the track')
    parser.add_argument('--artist', type=str, help='Artist of the track')
    parser.add_argument('--mood', type=str, help='Mood of the track')
    parser.add_argument('--genre', type=str, help='Genre of the track')
    parser.add_argument('--bpm', type=float, help='BPM (tempo) of the track')
    parser.add_argument('--tags', type=str, help='Comma-separated tags for the track')
    parser.add_argument('--description', type=str, help='Description of the track')
    parser.add_argument('--copyright-free', action='store_true', help='Mark track as copyright-free')
    
    # Collection arguments
    parser.add_argument('--collection-name', type=str, help='Name for new collection')
    parser.add_argument('--collection-description', type=str, help='Description for new collection')
    parser.add_argument('--collection-tags', type=str, help='Comma-separated tags for new collection')
    
    # Search arguments
    parser.add_argument('--search-mood', type=str, help='Mood to search for')
    parser.add_argument('--search-genre', type=str, help='Genre to search for')
    parser.add_argument('--search-bpm', type=float, help='BPM to search for')
    parser.add_argument('--search-duration', type=float, help='Duration to search for (seconds)')
    parser.add_argument('--search-keywords', type=str, help='Comma-separated keywords to search for')
    parser.add_argument('--search-copyright-free', action='store_true', help='Only search copyright-free tracks')
    parser.add_argument('--search-collection-id', type=str, help='Collection ID to search in')
    parser.add_argument('--max-results', type=int, default=10, help='Maximum number of search results')
    
    # View arguments
    parser.add_argument('--view-collections', action='store_true', help='Display all collections')
    parser.add_argument('--view-collection-id', type=str, help='Display tracks in a specific collection')
    
    # Mode selection
    parser.add_argument('--add-track', action='store_true', help='Add a track to the library')
    parser.add_argument('--create-collection', action='store_true', help='Create a new collection')
    parser.add_argument('--add-to-collection', action='store_true', help='Add tracks to a collection')
    parser.add_argument('--search', action='store_true', help='Search for tracks')
    
    args = parser.parse_args()
    
    # Configure the music library
    config = {
        'music_library_path': args.library_path,
        'ffmpeg_path': args.ffmpeg,
        'ffprobe_path': args.ffprobe
    }
    
    # Initialize the music library
    library = init_music_library(config)
    
    # Track the created IDs for later use
    created_track_ids = []
    created_collection_id = None
    
    # Execute requested operations
    if args.add_track:
        track_id = add_track_example(library, args)
        if track_id:
            created_track_ids.append(track_id)
    
    if args.create_collection:
        created_collection_id = create_collection_example(library, args)
    
    if args.add_to_collection:
        # If we created tracks and a collection in this run, add them to the collection
        if created_collection_id and created_track_ids:
            add_tracks_to_collection_example(library, created_collection_id, created_track_ids)
        # Otherwise, try to use provided collection ID
        elif args.search_collection_id and created_track_ids:
            add_tracks_to_collection_example(library, args.search_collection_id, created_track_ids)
        else:
            logger.warning("Missing collection ID or track IDs for adding to collection")
    
    if args.search:
        search_tracks_example(library, args)
    
    if args.view_collections:
        display_collections_example(library)
    
    if args.view_collection_id:
        display_collection_tracks_example(library, args.view_collection_id)
    elif created_collection_id:
        display_collection_tracks_example(library, created_collection_id)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 