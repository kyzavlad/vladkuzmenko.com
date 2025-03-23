"""
Music Recommendation Engine

This module provides a recommendation engine for selecting appropriate music
for video content based on multiple factors including mood, tempo, genre,
emotional arc, and user preferences. It combines information from all other
music-related components to provide intelligent recommendations.
"""

import os
import logging
import json
import random
import time
from typing import Dict, Any, List, Optional, Tuple, Set
from pathlib import Path
import numpy as np
from collections import Counter

logger = logging.getLogger(__name__)

class MusicRecommender:
    """
    Intelligent music recommendation engine for video content.
    
    This class integrates with all other music-related components to provide
    smart recommendations based on video content analysis and user preferences.
    It can learn from feedback to improve recommendations over time.
    
    Features:
    - Content-based filtering using video mood, tempo, genre
    - Collaborative filtering based on user ratings and feedback
    - Emotional arc matching for dynamic soundtracks
    - Cross-source recommendations (local library and external services)
    - Personalized recommendations based on user history
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the music recommendation engine.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        
        # Set default parameters
        self.preference_db_path = self.config.get(
            'preference_db_path', 
            os.path.join(os.path.expanduser('~'), '.music_recommender', 'preferences.json')
        )
        self.history_db_path = self.config.get(
            'history_db_path', 
            os.path.join(os.path.expanduser('~'), '.music_recommender', 'history.json')
        )
        self.feedback_db_path = self.config.get(
            'feedback_db_path', 
            os.path.join(os.path.expanduser('~'), '.music_recommender', 'feedback.json')
        )
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.preference_db_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.history_db_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.feedback_db_path), exist_ok=True)
        
        # Initialize databases
        self.preferences = self._load_database(self.preference_db_path)
        self.history = self._load_database(self.history_db_path)
        self.feedback = self._load_database(self.feedback_db_path)
        
        # Initialize components
        from app.services.music.music_selector import MusicSelector
        from app.services.music.music_library import MusicLibrary
        from app.services.music.external_music_service import ExternalMusicService
        
        self.music_selector = MusicSelector(config)
        self.music_library = MusicLibrary(config)
        self.external_music_service = ExternalMusicService(config) if self.config.get('use_external_services', True) else None
        
        # Learning rate for preference updates
        self.learning_rate = self.config.get('learning_rate', 0.2)
        
        # Recommendation weights
        self.weights = self.config.get('recommendation_weights', {
            'content_match': 0.6,
            'user_preference': 0.2,
            'popularity': 0.1,
            'diversity': 0.1
        })
    
    def recommend_for_video(
        self,
        video_path: str,
        user_id: Optional[str] = None,
        transcript: Optional[List[Dict[str, Any]]] = None,
        target_duration: Optional[float] = None,
        mood_override: Optional[str] = None,
        tempo_override: Optional[float] = None,
        genre_override: Optional[str] = None,
        max_results: int = 10,
        copyright_free_only: bool = False,
        collection_id: Optional[str] = None,
        use_emotional_arc: bool = True,
        include_external_services: bool = True,
        segment_duration: Optional[int] = None,
        diversity_level: float = 0.3
    ) -> Dict[str, Any]:
        """
        Recommend music tracks for a specific video.
        
        This method analyzes the video content and provides personalized music
        recommendations based on content analysis and user preferences.
        
        Args:
            video_path: Path to the video file
            user_id: Optional user ID for personalized recommendations
            transcript: Optional transcript data
            target_duration: Optional target duration for music
            mood_override: Optional mood override
            tempo_override: Optional tempo override
            genre_override: Optional genre override
            max_results: Maximum number of recommendations to return
            copyright_free_only: Only return copyright-free tracks
            collection_id: Optional collection ID to search within
            use_emotional_arc: Whether to use emotional arc analysis
            include_external_services: Whether to include external services
            segment_duration: Duration of segments for timeline analysis
            diversity_level: Level of diversity in recommendations (0-1)
            
        Returns:
            Dictionary with recommendation results
        """
        results = {
            "status": "success",
            "input_path": video_path,
            "recommendations": [],
            "emotional_timeline": None
        }
        
        try:
            # Step 1: Get content-based recommendations using MusicSelector
            content_results = self.music_selector.select_music(
                video_path=video_path,
                transcript=transcript,
                target_duration=target_duration,
                mood_override=mood_override,
                tempo_override=tempo_override,
                genre_override=genre_override,
                max_results=max_results * 2,  # Get more results for filtering
                copyright_free_only=copyright_free_only,
                collection_id=collection_id,
                use_emotional_arc=use_emotional_arc,
                segment_duration=segment_duration
            )
            
            # Extract content-based tracks
            content_tracks = content_results.get("recommendations", [])
            mood_analysis = content_results.get("mood_analysis")
            emotional_arc = content_results.get("emotional_arc")
            
            # Extract mood and genre if available
            detected_mood = None
            detected_genre = None
            
            if mood_analysis and "overall_mood" in mood_analysis and "mood_labels" in mood_analysis["overall_mood"]:
                mood_labels = mood_analysis["overall_mood"]["mood_labels"]
                if mood_labels:
                    detected_mood = mood_labels[0]
            
            if genre_override:
                detected_genre = genre_override
            elif content_tracks and "genre" in content_tracks[0]:
                # Use genre from top recommendation
                detected_genre = content_tracks[0]["genre"]
            
            # Step 2: Get emotional timeline if available
            emotional_timeline = None
            if emotional_arc and use_emotional_arc:
                # Use emotional arc for segment-based recommendations
                emotional_arc_results = self.music_selector.select_music_for_emotional_arc(
                    video_path=video_path,
                    transcript=transcript,
                    segment_duration=segment_duration or 5,
                    copyright_free_only=copyright_free_only,
                    collection_id=collection_id
                )
                
                emotional_timeline = emotional_arc_results.get("music_timeline")
                timeline_tracks = emotional_arc_results.get("segment_tracks", [])
                
                # Add timeline tracks to content tracks
                for track in timeline_tracks:
                    if isinstance(track, dict) and "track_id" in track:
                        content_tracks.append(track)
                
                results["emotional_timeline"] = emotional_timeline
            
            # Step 3: Get user preference-based recommendations
            preference_tracks = []
            if user_id and detected_mood:
                # Get user preferences based on mood and genre
                preference_results = self._get_preferences_based_recommendations(
                    user_id=user_id,
                    mood=detected_mood,
                    genre=detected_genre,
                    max_results=max_results
                )
                preference_tracks = preference_results.get("tracks", [])
            
            # Step 4: Get external service recommendations if requested
            external_tracks = []
            if include_external_services and self.external_music_service is not None:
                external_results = self._get_external_recommendations(
                    mood=detected_mood,
                    genre=detected_genre,
                    max_results=max_results
                )
                external_tracks = external_results.get("tracks", [])
            
            # Step 5: Combine and rank all recommendations
            all_tracks = self._combine_recommendations(
                content_tracks=content_tracks,
                preference_tracks=preference_tracks,
                external_tracks=external_tracks,
                user_id=user_id,
                detected_mood=detected_mood,
                detected_genre=detected_genre,
                diversity_level=diversity_level
            )
            
            # Limit to max_results
            results["recommendations"] = all_tracks[:max_results]
            results["total_matches"] = len(all_tracks)
            
            # Include analysis results
            if mood_analysis:
                results["mood_analysis"] = {
                    "mood": detected_mood,
                    "valence": mood_analysis.get("valence_arousal", {}).get("valence", 0.5),
                    "arousal": mood_analysis.get("valence_arousal", {}).get("arousal", 0.5)
                }
            
            # Add recommendation to history if user_id provided
            if user_id:
                self._add_to_history(
                    user_id=user_id,
                    video_path=video_path,
                    recommendations=results["recommendations"],
                    mood=detected_mood,
                    genre=detected_genre
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Error recommending music: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def recommend_similar_to_track(
        self,
        track_id: str,
        user_id: Optional[str] = None,
        source: str = "library",
        max_results: int = 10,
        include_external_services: bool = True
    ) -> Dict[str, Any]:
        """
        Recommend tracks similar to a specific track.
        
        Args:
            track_id: ID of the reference track
            user_id: Optional user ID for personalized recommendations
            source: Source of the track ('library' or external service name)
            max_results: Maximum number of recommendations to return
            include_external_services: Whether to include external services
            
        Returns:
            Dictionary with recommendation results
        """
        results = {
            "status": "success",
            "reference_track_id": track_id,
            "reference_source": source,
            "recommendations": []
        }
        
        try:
            # Step 1: Get reference track details
            track_details = None
            if source == "library":
                # Get track from local library
                track_response = self.music_library.get_track(track_id)
                if track_response.get("status") == "success":
                    track_details = track_response
            else:
                # Get track from external service
                if self.external_music_service:
                    track_info = self.external_music_service._get_track_info(track_id, source)
                    if track_info.get("status") == "success":
                        track_details = track_info
            
            if not track_details:
                return {
                    "status": "error",
                    "error": f"Could not find track {track_id} in {source}"
                }
            
            # Extract track attributes
            mood = track_details.get("mood", "neutral")
            genre = track_details.get("genre", "unknown")
            bpm = track_details.get("bpm")
            
            # Step 2: Get similar tracks from local library
            library_tracks = []
            library_results = self.music_library.search_tracks(
                mood=mood,
                genre=genre,
                tempo=bpm,
                max_results=max_results * 2
            )
            library_tracks = library_results.get("tracks", [])
            
            # Remove the reference track from results
            library_tracks = [t for t in library_tracks if t.get("track_id") != track_id]
            
            # Step 3: Get similar tracks from external services if requested
            external_tracks = []
            if include_external_services and self.external_music_service is not None:
                query = f"{track_details.get('title', '')} {track_details.get('artist', '')}"
                external_results = self.external_music_service.search_tracks(
                    query=query,
                    mood=mood,
                    genre=genre,
                    bpm=bpm,
                    max_results=max_results
                )
                external_tracks = external_results.get("tracks", [])
            
            # Step 4: Combine and rank all recommendations
            all_tracks = self._combine_recommendations(
                content_tracks=library_tracks,
                preference_tracks=[],
                external_tracks=external_tracks,
                user_id=user_id,
                detected_mood=mood,
                detected_genre=genre
            )
            
            # Limit to max_results
            results["recommendations"] = all_tracks[:max_results]
            results["total_matches"] = len(all_tracks)
            results["reference_track"] = {
                "title": track_details.get("title"),
                "artist": track_details.get("artist"),
                "mood": mood,
                "genre": genre
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding similar tracks: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def submit_feedback(
        self,
        user_id: str,
        track_id: str,
        source: str = "library",
        rating: Optional[int] = None,
        liked: Optional[bool] = None,
        used_in_project: Optional[bool] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Submit user feedback for a track recommendation.
        
        This method records user feedback and updates user preferences to
        improve future recommendations.
        
        Args:
            user_id: ID of the user submitting feedback
            track_id: ID of the track being rated
            source: Source of the track ('library' or external service name)
            rating: Optional rating (1-5)
            liked: Optional boolean indicating if the user liked the track
            used_in_project: Optional boolean indicating if the track was used
            context: Optional context information (video ID, project ID, etc.)
            
        Returns:
            Dictionary with feedback submission results
        """
        results = {
            "status": "success",
            "user_id": user_id,
            "track_id": track_id,
            "source": source
        }
        
        try:
            # Get track details
            track_details = None
            if source == "library":
                # Get track from local library
                track_response = self.music_library.get_track(track_id)
                if track_response.get("status") == "success":
                    track_details = track_response
            else:
                # Get track from external service
                if self.external_music_service:
                    track_info = self.external_music_service._get_track_info(track_id, source)
                    if track_info.get("status") == "success":
                        track_details = track_info
            
            if not track_details:
                return {
                    "status": "error",
                    "error": f"Could not find track {track_id} in {source}"
                }
            
            # Create feedback entry
            feedback_entry = {
                "user_id": user_id,
                "track_id": track_id,
                "source": source,
                "timestamp": time.time()
            }
            
            if rating is not None:
                feedback_entry["rating"] = max(1, min(5, rating))
            
            if liked is not None:
                feedback_entry["liked"] = liked
            
            if used_in_project is not None:
                feedback_entry["used_in_project"] = used_in_project
            
            if context is not None:
                feedback_entry["context"] = context
            
            # Add feedback to database
            if "feedback" not in self.feedback:
                self.feedback["feedback"] = []
            
            self.feedback["feedback"].append(feedback_entry)
            self._save_database(self.feedback_db_path, self.feedback)
            
            # Update user preferences based on feedback
            if user_id not in self.preferences:
                self.preferences[user_id] = {
                    "moods": {},
                    "genres": {},
                    "artists": {},
                    "tempo_range": [60, 120],
                    "favorite_tracks": []
                }
            
            user_prefs = self.preferences[user_id]
            
            # Extract attributes from track
            mood = track_details.get("mood")
            genre = track_details.get("genre")
            artist = track_details.get("artist")
            
            # Update mood preferences
            if mood and (rating is not None or liked is not None):
                if "moods" not in user_prefs:
                    user_prefs["moods"] = {}
                
                if mood not in user_prefs["moods"]:
                    user_prefs["moods"][mood] = 0.5  # Default neutral preference
                
                # Calculate update value
                update_value = 0
                if rating is not None:
                    # Map rating 1-5 to preference update -0.2 to 0.2
                    update_value = (rating - 3) * self.learning_rate / 2
                elif liked is not None:
                    # Map liked True/False to preference update 0.2/-0.2
                    update_value = self.learning_rate if liked else -self.learning_rate
                
                # Update preference
                current_value = user_prefs["moods"][mood]
                user_prefs["moods"][mood] = max(0, min(1, current_value + update_value))
            
            # Update genre preferences
            if genre and (rating is not None or liked is not None):
                if "genres" not in user_prefs:
                    user_prefs["genres"] = {}
                
                if genre not in user_prefs["genres"]:
                    user_prefs["genres"][genre] = 0.5  # Default neutral preference
                
                # Calculate update value
                update_value = 0
                if rating is not None:
                    # Map rating 1-5 to preference update -0.2 to 0.2
                    update_value = (rating - 3) * self.learning_rate / 2
                elif liked is not None:
                    # Map liked True/False to preference update 0.2/-0.2
                    update_value = self.learning_rate if liked else -self.learning_rate
                
                # Update preference
                current_value = user_prefs["genres"][genre]
                user_prefs["genres"][genre] = max(0, min(1, current_value + update_value))
            
            # Update artist preferences
            if artist and (rating is not None or liked is not None):
                if "artists" not in user_prefs:
                    user_prefs["artists"] = {}
                
                if artist not in user_prefs["artists"]:
                    user_prefs["artists"][artist] = 0.5  # Default neutral preference
                
                # Calculate update value
                update_value = 0
                if rating is not None:
                    # Map rating 1-5 to preference update -0.2 to 0.2
                    update_value = (rating - 3) * self.learning_rate / 2
                elif liked is not None:
                    # Map liked True/False to preference update 0.2/-0.2
                    update_value = self.learning_rate if liked else -self.learning_rate
                
                # Update preference
                current_value = user_prefs["artists"][artist]
                user_prefs["artists"][artist] = max(0, min(1, current_value + update_value))
            
            # Update favorite tracks
            if "favorite_tracks" not in user_prefs:
                user_prefs["favorite_tracks"] = []
            
            # Add to favorites if highly rated or liked
            if (rating is not None and rating >= 4) or (liked is not None and liked):
                # Add if not already in favorites
                track_entry = {
                    "track_id": track_id,
                    "source": source
                }
                
                if track_entry not in user_prefs["favorite_tracks"]:
                    user_prefs["favorite_tracks"].append(track_entry)
            
            # Remove from favorites if low rated or disliked
            if (rating is not None and rating <= 2) or (liked is not None and not liked):
                # Remove if in favorites
                user_prefs["favorite_tracks"] = [
                    t for t in user_prefs["favorite_tracks"]
                    if not (t["track_id"] == track_id and t["source"] == source)
                ]
            
            # Save updated preferences
            self._save_database(self.preference_db_path, self.preferences)
            
            results["preferences_updated"] = True
            
            return results
            
        except Exception as e:
            logger.error(f"Error submitting feedback: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_user_preferences(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Get user preferences for music recommendations.
        
        Args:
            user_id: ID of the user
            
        Returns:
            Dictionary with user preferences
        """
        if user_id in self.preferences:
            return {
                "status": "success",
                "user_id": user_id,
                "preferences": self.preferences[user_id]
            }
        else:
            return {
                "status": "error",
                "error": f"No preferences found for user {user_id}"
            }
    
    def get_user_recommendation_history(
        self,
        user_id: str,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Get history of recommendations for a user.
        
        Args:
            user_id: ID of the user
            limit: Maximum number of history entries to return
            
        Returns:
            Dictionary with user recommendation history
        """
        if user_id in self.history:
            history_entries = self.history[user_id]
            
            # Sort by timestamp (newest first)
            history_entries.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            
            # Limit number of entries
            limited_entries = history_entries[:limit]
            
            return {
                "status": "success",
                "user_id": user_id,
                "history": limited_entries,
                "total_entries": len(history_entries)
            }
        else:
            return {
                "status": "error",
                "error": f"No history found for user {user_id}"
            }
    
    def _get_preferences_based_recommendations(
        self,
        user_id: str,
        mood: Optional[str] = None,
        genre: Optional[str] = None,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Get recommendations based on user preferences.
        
        Args:
            user_id: ID of the user
            mood: Optional mood to filter by
            genre: Optional genre to filter by
            max_results: Maximum number of recommendations to return
            
        Returns:
            Dictionary with recommendation results
        """
        results = {
            "status": "success",
            "tracks": []
        }
        
        try:
            # Check if user preferences exist
            if user_id not in self.preferences:
                return results  # Return empty results
            
            user_prefs = self.preferences[user_id]
            
            # Get favorite tracks
            favorite_tracks = []
            if "favorite_tracks" in user_prefs:
                for track_entry in user_prefs.get("favorite_tracks", []):
                    track_id = track_entry.get("track_id")
                    source = track_entry.get("source", "library")
                    
                    track_details = None
                    if source == "library":
                        # Get track from local library
                        track_response = self.music_library.get_track(track_id)
                        if track_response.get("status") == "success":
                            track_details = track_response
                    
                    if track_details:
                        favorite_tracks.append(track_details)
            
            # Get preferred moods
            preferred_moods = []
            if "moods" in user_prefs:
                # Sort moods by preference score (descending)
                sorted_moods = sorted(
                    user_prefs["moods"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # Get moods with preference score > 0.6
                preferred_moods = [mood for mood, score in sorted_moods if score > 0.6]
            
            # Get preferred genres
            preferred_genres = []
            if "genres" in user_prefs:
                # Sort genres by preference score (descending)
                sorted_genres = sorted(
                    user_prefs["genres"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # Get genres with preference score > 0.6
                preferred_genres = [genre for genre, score in sorted_genres if score > 0.6]
            
            # Get preferred artists
            preferred_artists = []
            if "artists" in user_prefs:
                # Sort artists by preference score (descending)
                sorted_artists = sorted(
                    user_prefs["artists"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # Get artists with preference score > 0.6
                preferred_artists = [artist for artist, score in sorted_artists if score > 0.6]
            
            # Search tracks based on preferences
            search_params = {
                "max_results": max_results * 2
            }
            
            # Add mood parameter
            if mood and mood in user_prefs.get("moods", {}):
                # If the detected mood is in user preferences, use it
                search_params["mood"] = mood
            elif preferred_moods:
                # Otherwise use the top preferred mood
                search_params["mood"] = preferred_moods[0]
            
            # Add genre parameter
            if genre and genre in user_prefs.get("genres", {}):
                # If the detected genre is in user preferences, use it
                search_params["genre"] = genre
            elif preferred_genres:
                # Otherwise use the top preferred genre
                search_params["genre"] = preferred_genres[0]
            
            # Search for tracks
            search_results = self.music_library.search_tracks(**search_params)
            preference_tracks = search_results.get("tracks", [])
            
            # Filter tracks by preferred artists
            if preferred_artists:
                for track in preference_tracks:
                    artist = track.get("artist")
                    if artist in preferred_artists:
                        track["preference_score"] = 1.0
                    else:
                        track["preference_score"] = 0.5
            
            results["tracks"] = preference_tracks
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting preference-based recommendations: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _get_external_recommendations(
        self,
        mood: Optional[str] = None,
        genre: Optional[str] = None,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Get recommendations from external music services.
        
        Args:
            mood: Optional mood to filter by
            genre: Optional genre to filter by
            max_results: Maximum number of recommendations to return
            
        Returns:
            Dictionary with recommendation results
        """
        results = {
            "status": "success",
            "tracks": []
        }
        
        try:
            # Check if external music service is available
            if self.external_music_service is None:
                return results  # Return empty results
            
            # Build search query
            query = ""
            if mood:
                query += f"{mood} "
            if genre:
                query += f"{genre} "
            
            if not query:
                query = "music"
            
            # Search for tracks
            search_results = self.external_music_service.search_tracks(
                query=query,
                mood=mood,
                genre=genre,
                max_results=max_results,
                license_type="cc"  # Default to Creative Commons
            )
            
            # Add tracks to results
            results["tracks"] = search_results.get("tracks", [])
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting external recommendations: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _combine_recommendations(
        self,
        content_tracks: List[Dict[str, Any]],
        preference_tracks: List[Dict[str, Any]],
        external_tracks: List[Dict[str, Any]],
        user_id: Optional[str] = None,
        detected_mood: Optional[str] = None,
        detected_genre: Optional[str] = None,
        diversity_level: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Combine and rank recommendations from different sources.
        
        Args:
            content_tracks: Tracks from content-based selection
            preference_tracks: Tracks from user preferences
            external_tracks: Tracks from external services
            user_id: Optional user ID
            detected_mood: Detected mood of the video
            detected_genre: Detected genre of the video
            diversity_level: Level of diversity in recommendations (0-1)
            
        Returns:
            Combined and ranked list of tracks
        """
        # Combine all tracks
        all_tracks = []
        
        # Function to check if a track is already in the list
        def track_exists(track_id, source, tracks_list):
            for track in tracks_list:
                if track.get("track_id") == track_id and track.get("source", "library") == source:
                    return True
            return False
        
        # Add content tracks
        for track in content_tracks:
            track_id = track.get("track_id")
            source = track.get("source", "library")
            
            if not track_exists(track_id, source, all_tracks):
                # Add score based on content match
                if "relevance_score" not in track:
                    track["relevance_score"] = 0.7
                
                # Set source type
                track["recommendation_source"] = "content"
                
                all_tracks.append(track)
        
        # Add preference tracks
        for track in preference_tracks:
            track_id = track.get("track_id")
            source = track.get("source", "library")
            
            if not track_exists(track_id, source, all_tracks):
                # Add score based on preference match
                if "preference_score" not in track:
                    track["preference_score"] = 0.7
                
                # Set source type
                track["recommendation_source"] = "preference"
                
                all_tracks.append(track)
            else:
                # Update existing track with preference score
                for existing_track in all_tracks:
                    if (existing_track.get("track_id") == track_id and 
                        existing_track.get("source", "library") == source):
                        existing_track["preference_score"] = track.get("preference_score", 0.7)
                        if "recommendation_source" not in existing_track:
                            existing_track["recommendation_source"] = "preference+content"
        
        # Add external tracks
        for track in external_tracks:
            track_id = track.get("track_id")
            source = track.get("source", "library")
            
            if not track_exists(track_id, source, all_tracks):
                # Add score based on external match
                if "relevance_score" not in track:
                    track["relevance_score"] = 0.6
                
                # Set source type
                track["recommendation_source"] = "external"
                
                all_tracks.append(track)
        
        # Calculate diversity-based scores
        if detected_mood and detected_genre and diversity_level > 0:
            # Count mood and genre occurrences
            mood_counts = Counter([t.get("mood") for t in all_tracks if "mood" in t])
            genre_counts = Counter([t.get("genre") for t in all_tracks if "genre" in t])
            
            # Calculate diversity scores
            for track in all_tracks:
                mood = track.get("mood")
                genre = track.get("genre")
                
                diversity_score = 1.0
                
                if mood and mood in mood_counts and mood_counts[mood] > 1:
                    # Reduce score for common moods
                    mood_diversity = 1.0 - (mood_counts[mood] / len(all_tracks)) * diversity_level
                    diversity_score *= mood_diversity
                
                if genre and genre in genre_counts and genre_counts[genre] > 1:
                    # Reduce score for common genres
                    genre_diversity = 1.0 - (genre_counts[genre] / len(all_tracks)) * diversity_level
                    diversity_score *= genre_diversity
                
                track["diversity_score"] = diversity_score
        
        # Calculate final scores and rank tracks
        for track in all_tracks:
            # Initial score components
            content_score = track.get("relevance_score", 0.0) * self.weights["content_match"]
            preference_score = track.get("preference_score", 0.0) * self.weights["user_preference"]
            diversity_score = track.get("diversity_score", 1.0) * self.weights["diversity"]
            
            # Popularity boost
            popularity_score = 0.0
            if "playcount" in track:
                # Normalize playcount (0-1)
                max_playcount = 10000  # Arbitrary maximum for normalization
                normalized_playcount = min(1.0, track["playcount"] / max_playcount)
                popularity_score = normalized_playcount * self.weights["popularity"]
            
            # Calculate final score
            final_score = content_score + preference_score + diversity_score + popularity_score
            
            # Boost copyright-free tracks slightly
            if track.get("copyright_free", False):
                final_score += 0.05
            
            track["final_score"] = final_score
        
        # Sort by final score (descending)
        all_tracks.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        
        return all_tracks
    
    def _add_to_history(
        self,
        user_id: str,
        video_path: str,
        recommendations: List[Dict[str, Any]],
        mood: Optional[str] = None,
        genre: Optional[str] = None
    ) -> None:
        """
        Add a recommendation to the user's history.
        
        Args:
            user_id: ID of the user
            video_path: Path to the video
            recommendations: List of recommended tracks
            mood: Detected mood of the video
            genre: Detected genre of the video
        """
        try:
            # Create history entry
            history_entry = {
                "video_path": video_path,
                "timestamp": time.time(),
                "recommendations": recommendations,
                "mood": mood,
                "genre": genre
            }
            
            # Add to user's history
            if user_id not in self.history:
                self.history[user_id] = []
            
            self.history[user_id].append(history_entry)
            
            # Limit history size (keep last 50 entries)
            if len(self.history[user_id]) > 50:
                self.history[user_id] = self.history[user_id][-50:]
            
            # Save history
            self._save_database(self.history_db_path, self.history)
            
        except Exception as e:
            logger.error(f"Error adding to history: {str(e)}")
    
    def _load_database(self, db_path: str) -> Dict[str, Any]:
        """
        Load a database from a JSON file.
        
        Args:
            db_path: Path to the database file
            
        Returns:
            Database dictionary
        """
        if os.path.exists(db_path):
            try:
                with open(db_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading database from {db_path}: {str(e)}")
        
        return {}
    
    def _save_database(self, db_path: str, data: Dict[str, Any]) -> None:
        """
        Save a database to a JSON file.
        
        Args:
            db_path: Path to the database file
            data: Database dictionary
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            with open(db_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving database to {db_path}: {str(e)}") 