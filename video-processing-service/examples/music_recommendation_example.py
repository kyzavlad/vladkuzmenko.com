#!/usr/bin/env python3
"""
Music Recommendation Engine Example

This script demonstrates the usage of the MusicRecommender class for recommending
music tracks for videos and getting similar track recommendations based on a reference track.
It also shows how to submit user feedback and retrieve user preferences.
"""

import os
import sys
import argparse
import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.services.music.music_recommender import MusicRecommender
from app.services.music.music_library import MusicLibrary

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_recommender(config: Optional[Dict[str, Any]] = None) -> MusicRecommender:
    """Initialize the music recommender with configuration."""
    logger.info("Initializing music recommender...")
    return MusicRecommender(config)

def recommend_for_video_example(
    recommender: MusicRecommender,
    video_path: str,
    user_id: Optional[str] = None,
    transcript_path: Optional[str] = None,
    mood_override: Optional[str] = None,
    genre_override: Optional[str] = None,
    tempo_override: Optional[float] = None,
    max_results: int = 5,
    copyright_free_only: bool = False,
    use_emotional_arc: bool = True
) -> Dict[str, Any]:
    """
    Demonstrate recommending music for a video.
    
    Args:
        recommender: MusicRecommender instance
        video_path: Path to the video file
        user_id: Optional user ID for personalized recommendations
        transcript_path: Optional path to transcript file
        mood_override: Optional mood override
        genre_override: Optional genre override
        tempo_override: Optional tempo override
        max_results: Maximum number of recommendations
        copyright_free_only: Only return copyright-free tracks
        use_emotional_arc: Whether to use emotional arc analysis
        
    Returns:
        Recommendation results
    """
    logger.info(f"Recommending music for video: {video_path}")
    
    # Load transcript if provided
    transcript = None
    if transcript_path and os.path.exists(transcript_path):
        with open(transcript_path, 'r') as f:
            transcript = json.load(f)
    
    # Get recommendations
    results = recommender.recommend_for_video(
        video_path=video_path,
        user_id=user_id,
        transcript=transcript,
        mood_override=mood_override,
        genre_override=genre_override,
        tempo_override=tempo_override,
        max_results=max_results,
        copyright_free_only=copyright_free_only,
        use_emotional_arc=use_emotional_arc
    )
    
    # Check if recommendations were successful
    if results.get("status") == "success":
        recommendations = results.get("recommendations", [])
        
        logger.info(f"Found {len(recommendations)} recommendations:")
        
        for i, track in enumerate(recommendations, 1):
            logger.info(f"Recommendation {i}:")
            logger.info(f"  Track ID: {track.get('track_id')}")
            logger.info(f"  Title: {track.get('title', 'Unknown')}")
            logger.info(f"  Artist: {track.get('artist', 'Unknown')}")
            logger.info(f"  Mood: {track.get('mood', 'Unknown')}")
            logger.info(f"  Genre: {track.get('genre', 'Unknown')}")
            logger.info(f"  Score: {track.get('final_score', 0.0):.2f}")
            logger.info(f"  Source: {track.get('recommendation_source', 'Unknown')}")
            logger.info("")
        
        # Check if emotional timeline is available
        if results.get("emotional_timeline"):
            timeline = results.get("emotional_timeline")
            logger.info(f"Emotional timeline available with {len(timeline)} segments")
            
            for i, segment in enumerate(timeline, 1):
                logger.info(f"Segment {i}:")
                logger.info(f"  Start: {segment.get('start_time', 0):.2f}s")
                logger.info(f"  End: {segment.get('end_time', 0):.2f}s")
                logger.info(f"  Duration: {segment.get('duration', 0):.2f}s")
                logger.info(f"  Mood: {segment.get('mood', 'Unknown')}")
                logger.info(f"  Cue Type: {segment.get('cue_type', 'Unknown')}")
                logger.info("")
    else:
        logger.error(f"Error getting recommendations: {results.get('error', 'Unknown error')}")
    
    return results

def recommend_similar_tracks_example(
    recommender: MusicRecommender,
    track_id: str,
    source: str = "library",
    user_id: Optional[str] = None,
    max_results: int = 5
) -> Dict[str, Any]:
    """
    Demonstrate recommending tracks similar to a reference track.
    
    Args:
        recommender: MusicRecommender instance
        track_id: ID of the reference track
        source: Source of the track ('library' or external service name)
        user_id: Optional user ID for personalized recommendations
        max_results: Maximum number of recommendations
        
    Returns:
        Recommendation results
    """
    logger.info(f"Finding tracks similar to {track_id} from {source}")
    
    # Get similar track recommendations
    results = recommender.recommend_similar_to_track(
        track_id=track_id,
        user_id=user_id,
        source=source,
        max_results=max_results
    )
    
    # Check if recommendations were successful
    if results.get("status") == "success":
        reference_track = results.get("reference_track", {})
        recommendations = results.get("recommendations", [])
        
        logger.info(f"Reference track: {reference_track.get('title')} by {reference_track.get('artist')}")
        logger.info(f"Found {len(recommendations)} similar tracks:")
        
        for i, track in enumerate(recommendations, 1):
            logger.info(f"Similar Track {i}:")
            logger.info(f"  Track ID: {track.get('track_id')}")
            logger.info(f"  Title: {track.get('title', 'Unknown')}")
            logger.info(f"  Artist: {track.get('artist', 'Unknown')}")
            logger.info(f"  Mood: {track.get('mood', 'Unknown')}")
            logger.info(f"  Genre: {track.get('genre', 'Unknown')}")
            logger.info(f"  Score: {track.get('final_score', 0.0):.2f}")
            logger.info("")
    else:
        logger.error(f"Error finding similar tracks: {results.get('error', 'Unknown error')}")
    
    return results

def submit_feedback_example(
    recommender: MusicRecommender,
    user_id: str,
    track_id: str,
    rating: int = 5,
    liked: bool = True,
    used_in_project: bool = True
) -> Dict[str, Any]:
    """
    Demonstrate submitting feedback for a recommended track.
    
    Args:
        recommender: MusicRecommender instance
        user_id: ID of the user submitting feedback
        track_id: ID of the track being rated
        rating: Rating (1-5)
        liked: Whether the user liked the track
        used_in_project: Whether the track was used in a project
        
    Returns:
        Feedback submission results
    """
    logger.info(f"Submitting feedback for track {track_id}")
    logger.info(f"  User ID: {user_id}")
    logger.info(f"  Rating: {rating}/5")
    logger.info(f"  Liked: {liked}")
    logger.info(f"  Used in project: {used_in_project}")
    
    # Submit feedback
    results = recommender.submit_feedback(
        user_id=user_id,
        track_id=track_id,
        rating=rating,
        liked=liked,
        used_in_project=used_in_project,
        context={"source": "example_script"}
    )
    
    # Check if feedback submission was successful
    if results.get("status") == "success":
        logger.info("Feedback submitted successfully")
        logger.info(f"Preferences updated: {results.get('preferences_updated', False)}")
    else:
        logger.error(f"Error submitting feedback: {results.get('error', 'Unknown error')}")
    
    return results

def get_user_preferences_example(
    recommender: MusicRecommender,
    user_id: str
) -> Dict[str, Any]:
    """
    Demonstrate getting user preferences.
    
    Args:
        recommender: MusicRecommender instance
        user_id: ID of the user
        
    Returns:
        User preferences
    """
    logger.info(f"Getting preferences for user {user_id}")
    
    # Get user preferences
    results = recommender.get_user_preferences(user_id)
    
    # Check if preferences were retrieved successfully
    if results.get("status") == "success":
        preferences = results.get("preferences", {})
        
        logger.info("User preferences:")
        
        # Display mood preferences
        if "moods" in preferences:
            logger.info("  Mood preferences:")
            for mood, score in sorted(preferences["moods"].items(), key=lambda x: x[1], reverse=True):
                logger.info(f"    {mood}: {score:.2f}")
        
        # Display genre preferences
        if "genres" in preferences:
            logger.info("  Genre preferences:")
            for genre, score in sorted(preferences["genres"].items(), key=lambda x: x[1], reverse=True):
                logger.info(f"    {genre}: {score:.2f}")
        
        # Display artist preferences
        if "artists" in preferences:
            logger.info("  Artist preferences:")
            for artist, score in sorted(preferences["artists"].items(), key=lambda x: x[1], reverse=True):
                logger.info(f"    {artist}: {score:.2f}")
        
        # Display favorite tracks
        if "favorite_tracks" in preferences:
            logger.info(f"  Favorite tracks: {len(preferences['favorite_tracks'])}")
            for track in preferences["favorite_tracks"]:
                logger.info(f"    {track.get('track_id')} from {track.get('source', 'library')}")
    else:
        logger.error(f"Error getting user preferences: {results.get('error', 'Unknown error')}")
    
    return results

def get_track_from_library_example(music_library: MusicLibrary) -> Optional[str]:
    """
    Get a random track ID from the library for example purposes.
    
    Args:
        music_library: MusicLibrary instance
        
    Returns:
        Random track ID or None if no tracks found
    """
    # Get all tracks
    results = music_library.get_all_tracks()
    
    if results.get("status") == "success":
        tracks = results.get("tracks", [])
        
        if tracks:
            # Return the ID of the first track
            return tracks[0].get("track_id")
    
    return None

def main():
    """Main function to demonstrate music recommendation."""
    parser = argparse.ArgumentParser(description="Music recommendation example")
    
    # Operation selection
    parser.add_argument('--operation', choices=['video', 'similar', 'feedback', 'preferences', 'all'],
                      default='all', help='Operation to perform')
    
    # Video recommendation arguments
    parser.add_argument('--video-path', type=str, help='Path to the video file')
    parser.add_argument('--transcript-path', type=str, help='Path to the transcript file')
    parser.add_argument('--mood', type=str, help='Mood override')
    parser.add_argument('--genre', type=str, help='Genre override')
    
    # Similar track recommendation arguments
    parser.add_argument('--track-id', type=str, help='ID of the reference track')
    
    # User-related arguments
    parser.add_argument('--user-id', type=str, default='example_user', help='User ID')
    
    args = parser.parse_args()
    
    # Initialize the music recommender
    recommender = init_recommender()
    
    # Initialize music library
    music_library = MusicLibrary()
    
    # Get a random track ID if not specified
    track_id = args.track_id
    if not track_id and (args.operation == 'similar' or args.operation == 'all'):
        track_id = get_track_from_library_example(music_library)
        if track_id:
            logger.info(f"Using random track ID from library: {track_id}")
        else:
            logger.warning("No tracks found in library. Similar track example will be skipped.")
    
    # Run the requested operation
    if args.operation == 'video' or args.operation == 'all':
        if not args.video_path and args.operation == 'video':
            logger.error("Video recommendation requires --video-path")
            return
        
        if args.video_path and os.path.exists(args.video_path):
            recommend_for_video_example(
                recommender=recommender,
                video_path=args.video_path,
                user_id=args.user_id,
                transcript_path=args.transcript_path,
                mood_override=args.mood,
                genre_override=args.genre
            )
        elif args.operation == 'video':
            logger.error(f"Video file not found: {args.video_path}")
    
    if args.operation == 'similar' or args.operation == 'all':
        if not track_id and args.operation == 'similar':
            logger.error("Similar track recommendation requires --track-id")
            return
        
        if track_id:
            recommend_similar_tracks_example(
                recommender=recommender,
                track_id=track_id,
                user_id=args.user_id
            )
    
    if args.operation == 'feedback' or args.operation == 'all':
        if not track_id and args.operation == 'feedback':
            logger.error("Feedback submission requires --track-id")
            return
        
        if track_id:
            submit_feedback_example(
                recommender=recommender,
                user_id=args.user_id,
                track_id=track_id
            )
    
    if args.operation == 'preferences' or args.operation == 'all':
        get_user_preferences_example(
            recommender=recommender,
            user_id=args.user_id
        )

if __name__ == "__main__":
    main() 