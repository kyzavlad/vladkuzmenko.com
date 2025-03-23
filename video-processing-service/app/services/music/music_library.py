"""
Music Library Module

This module provides functionality for managing and accessing a library of
music tracks, including metadata and search capabilities.
"""

import os
import logging
import json
import random
import uuid
import time
import shutil
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class MusicLibrary:
    """
    Manages a library of music tracks.
    
    This class provides methods to search for and retrieve music tracks
    based on various criteria such as mood, tempo, genre, and duration.
    It also supports organizing tracks into collections for better management.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the music library.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        
        # Set default parameters
        self.library_path = self.config.get('music_library_path', os.path.join(os.getcwd(), 'music_library'))
        self.metadata_file = self.config.get('music_metadata_file', os.path.join(self.library_path, 'metadata.json'))
        self.collections_file = self.config.get('collections_metadata_file', os.path.join(self.library_path, 'collections.json'))
        
        # Create library directory if it doesn't exist
        os.makedirs(self.library_path, exist_ok=True)
        
        # Create subdirectories
        self.tracks_dir = os.path.join(self.library_path, 'tracks')
        os.makedirs(self.tracks_dir, exist_ok=True)
        
        # Load track metadata and collections
        self.tracks = self._load_metadata()
        self.collections = self._load_collections()
    
    def search_tracks(
        self,
        mood: Optional[str] = None,
        tempo: Optional[float] = None,
        genre: Optional[str] = None,
        duration: Optional[float] = None,
        keywords: Optional[List[str]] = None,
        max_results: int = 10,
        copyright_free_only: bool = False,
        collection_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search for tracks matching the specified criteria.
        
        Args:
            mood: Mood of the track
            tempo: Tempo (BPM) of the track
            genre: Genre of the track
            duration: Target duration in seconds
            keywords: Keywords to search for in track metadata
            max_results: Maximum number of results to return
            copyright_free_only: Only return copyright-free tracks
            collection_id: ID of collection to search in
            
        Returns:
            Dictionary with search results
        """
        results = {
            "status": "success",
            "query": {
                "mood": mood,
                "tempo": tempo,
                "genre": genre,
                "duration": duration,
                "keywords": keywords,
                "copyright_free_only": copyright_free_only,
                "collection_id": collection_id
            },
            "tracks": []
        }
        
        try:
            # Filter tracks based on criteria
            filtered_tracks = self.tracks
            
            # Filter by collection if specified
            if collection_id:
                collection = self.get_collection(collection_id)
                if not collection:
                    results["status"] = "error"
                    results["error"] = f"Collection not found: {collection_id}"
                    return results
                
                collection_track_ids = collection.get("track_ids", [])
                filtered_tracks = [t for t in filtered_tracks if t.get("id") in collection_track_ids]
            
            # Filter by copyright status if requested
            if copyright_free_only:
                filtered_tracks = [t for t in filtered_tracks if t.get("copyright_free", False)]
            
            # Filter by mood
            if mood:
                # First try exact match
                mood_tracks = [t for t in filtered_tracks if t.get("mood") == mood]
                # If no exact matches, try similar moods
                if not mood_tracks:
                    similar_moods = self._get_similar_moods(mood)
                    mood_tracks = [
                        t for t in filtered_tracks 
                        if t.get("mood") in similar_moods
                    ]
                filtered_tracks = mood_tracks
            
            # Filter by tempo (with tolerance)
            if tempo is not None:
                tempo_tolerance = self.config.get('tempo_tolerance', 10)  # BPM tolerance
                filtered_tracks = [
                    t for t in filtered_tracks 
                    if abs(t.get("bpm", 0) - tempo) <= tempo_tolerance
                ]
            
            # Filter by genre
            if genre:
                # First try exact match
                genre_tracks = [t for t in filtered_tracks if t.get("genre") == genre]
                # If no exact matches, try compatible genres
                if not genre_tracks:
                    compatible_genres = self._get_compatible_genres(genre)
                    genre_tracks = [
                        t for t in filtered_tracks 
                        if t.get("genre") in compatible_genres
                    ]
                filtered_tracks = genre_tracks
            
            # Filter by duration (with tolerance)
            if duration is not None:
                duration_tolerance = self.config.get('duration_tolerance', 30)  # seconds
                filtered_tracks = [
                    t for t in filtered_tracks 
                    if abs(t.get("duration", 0) - duration) <= duration_tolerance
                ]
            
            # Filter by keywords
            if keywords:
                keyword_tracks = []
                for track in filtered_tracks:
                    # Check if any keyword matches track metadata
                    for keyword in keywords:
                        keyword_lower = keyword.lower()
                        if (
                            keyword_lower in track.get("title", "").lower() or
                            keyword_lower in track.get("artist", "").lower() or
                            keyword_lower in track.get("description", "").lower() or
                            keyword_lower in " ".join(str(tag).lower() for tag in track.get("tags", []))
                        ):
                            keyword_tracks.append(track)
                            break
                filtered_tracks = keyword_tracks
            
            # If no tracks match the criteria, return empty results
            if not filtered_tracks:
                results["tracks"] = []
                results["total_matches"] = 0
                results["note"] = "No matching tracks found."
                return results
            
            # Calculate relevance scores and sort by relevance
            scored_tracks = []
            for track in filtered_tracks:
                score = self._calculate_relevance_score(
                    track, mood, tempo, genre, duration, keywords
                )
                scored_tracks.append((track, score))
            
            # Sort by score (descending)
            scored_tracks.sort(key=lambda x: x[1], reverse=True)
            
            # Extract tracks and add relevance scores
            top_tracks = []
            for track, score in scored_tracks[:max_results]:
                track_copy = track.copy()
                track_copy["relevance_score"] = score
                top_tracks.append(track_copy)
            
            results["tracks"] = top_tracks
            results["total_matches"] = len(filtered_tracks)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching tracks: {str(e)}")
            results["status"] = "error"
            results["error"] = str(e)
            return results
    
    def get_track(self, track_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a track by ID.
        
        Args:
            track_id: ID of the track
            
        Returns:
            Track metadata or None if not found
        """
        for track in self.tracks:
            if track.get("id") == track_id:
                return track
        return None
    
    def add_track(
        self,
        file_path: str,
        title: str,
        artist: str,
        mood: Optional[str] = None,
        genre: Optional[str] = None,
        bpm: Optional[float] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        copyright_free: bool = False,
        license: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add a track to the library.
        
        Args:
            file_path: Path to the audio file
            title: Title of the track
            artist: Artist of the track
            mood: Mood of the track
            genre: Genre of the track
            bpm: Tempo (BPM) of the track
            tags: Tags for the track
            description: Description of the track
            copyright_free: Whether the track is copyright-free
            license: License information for the track
            
        Returns:
            Dictionary with result of the operation
        """
        try:
            # Generate a unique ID for the track
            track_id = str(uuid.uuid4())
            
            # Get file information
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_name)[1]
            
            # Copy file to library
            dest_path = os.path.join(self.tracks_dir, f"{track_id}{file_ext}")
            shutil.copy2(file_path, dest_path)
            
            # Get duration and other audio properties
            duration = self._get_audio_duration(dest_path)
            
            # Create track metadata
            track = {
                "id": track_id,
                "title": title,
                "artist": artist,
                "file_path": dest_path,
                "file_name": file_name,
                "duration": duration,
                "copyright_free": copyright_free,
                "added_at": self._get_current_timestamp()
            }
            
            # Add optional metadata
            if mood:
                track["mood"] = mood
            if genre:
                track["genre"] = genre
            if bpm:
                track["bpm"] = bpm
            if tags:
                track["tags"] = tags
            if description:
                track["description"] = description
            if license:
                track["license"] = license
            
            # Add track to library
            self.tracks.append(track)
            
            # Save metadata
            self._save_metadata()
            
            return {
                "status": "success",
                "track_id": track_id,
                "message": f"Track '{title}' added to library"
            }
            
        except Exception as e:
            logger.error(f"Error adding track: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def remove_track(self, track_id: str) -> Dict[str, Any]:
        """
        Remove a track from the library.
        
        Args:
            track_id: ID of the track
            
        Returns:
            Dictionary with result of the operation
        """
        try:
            # Find track
            track = None
            for i, t in enumerate(self.tracks):
                if t.get("id") == track_id:
                    track = t
                    self.tracks.pop(i)
                    break
            
            if not track:
                return {
                    "status": "error",
                    "error": f"Track not found: {track_id}"
                }
            
            # Remove file
            file_path = track.get("file_path")
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
            
            # Remove from collections
            for collection in self.collections:
                track_ids = collection.get("track_ids", [])
                if track_id in track_ids:
                    track_ids.remove(track_id)
            
            # Save metadata
            self._save_metadata()
            self._save_collections()
            
            return {
                "status": "success",
                "message": f"Track '{track.get('title')}' removed from library"
            }
            
        except Exception as e:
            logger.error(f"Error removing track: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def update_track(
        self,
        track_id: str,
        title: Optional[str] = None,
        artist: Optional[str] = None,
        mood: Optional[str] = None,
        genre: Optional[str] = None,
        bpm: Optional[float] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        copyright_free: Optional[bool] = None,
        license: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update track metadata.
        
        Args:
            track_id: ID of the track
            title: New title of the track
            artist: New artist of the track
            mood: New mood of the track
            genre: New genre of the track
            bpm: New tempo (BPM) of the track
            tags: New tags for the track
            description: New description of the track
            copyright_free: Whether the track is copyright-free
            license: License information for the track
            
        Returns:
            Dictionary with result of the operation
        """
        try:
            # Find track
            track = None
            for t in self.tracks:
                if t.get("id") == track_id:
                    track = t
                    break
            
            if not track:
                return {
                    "status": "error",
                    "error": f"Track not found: {track_id}"
                }
            
            # Update metadata
            updated = False
            if title is not None:
                track["title"] = title
                updated = True
            if artist is not None:
                track["artist"] = artist
                updated = True
            if mood is not None:
                track["mood"] = mood
                updated = True
            if genre is not None:
                track["genre"] = genre
                updated = True
            if bpm is not None:
                track["bpm"] = bpm
                updated = True
            if tags is not None:
                track["tags"] = tags
                updated = True
            if description is not None:
                track["description"] = description
                updated = True
            if copyright_free is not None:
                track["copyright_free"] = copyright_free
                updated = True
            if license is not None:
                track["license"] = license
                updated = True
            
            # Save metadata if updated
            if updated:
                self._save_metadata()
            
            return {
                "status": "success",
                "message": f"Track '{track.get('title')}' updated"
            }
            
        except Exception as e:
            logger.error(f"Error updating track: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    # Collection management methods
    def create_collection(
        self,
        name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a new collection.
        
        Args:
            name: Name of the collection
            description: Description of the collection
            tags: Tags for the collection
            
        Returns:
            Dictionary with result of the operation
        """
        try:
            # Generate a unique ID for the collection
            collection_id = str(uuid.uuid4())
            
            # Create collection
            collection = {
                "id": collection_id,
                "name": name,
                "description": description,
                "tags": tags or [],
                "track_ids": [],
                "created_at": self._get_current_timestamp(),
                "updated_at": self._get_current_timestamp()
            }
            
            # Add collection
            self.collections.append(collection)
            
            # Save collections
            self._save_collections()
            
            return {
                "status": "success",
                "collection_id": collection_id,
                "message": f"Collection '{name}' created"
            }
            
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def update_collection(
        self,
        collection_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Update a collection.
        
        Args:
            collection_id: ID of the collection
            name: New name of the collection
            description: New description of the collection
            tags: New tags for the collection
            
        Returns:
            Dictionary with result of the operation
        """
        try:
            # Find collection
            collection = None
            for c in self.collections:
                if c.get("id") == collection_id:
                    collection = c
                    break
            
            if not collection:
                return {
                    "status": "error",
                    "error": f"Collection not found: {collection_id}"
                }
            
            # Update metadata
            updated = False
            if name is not None:
                collection["name"] = name
                updated = True
            if description is not None:
                collection["description"] = description
                updated = True
            if tags is not None:
                collection["tags"] = tags
                updated = True
            
            # Update timestamp if updated
            if updated:
                collection["updated_at"] = self._get_current_timestamp()
                self._save_collections()
            
            return {
                "status": "success",
                "message": f"Collection '{collection.get('name')}' updated"
            }
            
        except Exception as e:
            logger.error(f"Error updating collection: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def delete_collection(self, collection_id: str) -> Dict[str, Any]:
        """
        Delete a collection.
        
        Args:
            collection_id: ID of the collection
            
        Returns:
            Dictionary with result of the operation
        """
        try:
            # Find collection
            collection = None
            for i, c in enumerate(self.collections):
                if c.get("id") == collection_id:
                    collection = c
                    self.collections.pop(i)
                    break
            
            if not collection:
                return {
                    "status": "error",
                    "error": f"Collection not found: {collection_id}"
                }
            
            # Save collections
            self._save_collections()
            
            return {
                "status": "success",
                "message": f"Collection '{collection.get('name')}' deleted"
            }
            
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def add_tracks_to_collection(
        self,
        collection_id: str,
        track_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Add tracks to a collection.
        
        Args:
            collection_id: ID of the collection
            track_ids: IDs of tracks to add
            
        Returns:
            Dictionary with result of the operation
        """
        try:
            # Find collection
            collection = None
            for c in self.collections:
                if c.get("id") == collection_id:
                    collection = c
                    break
            
            if not collection:
                return {
                    "status": "error",
                    "error": f"Collection not found: {collection_id}"
                }
            
            # Add tracks
            added_tracks = []
            existing_tracks = []
            
            for track_id in track_ids:
                # Check if track exists
                track = self.get_track(track_id)
                if not track:
                    continue
                
                # Check if track is already in collection
                if track_id in collection.get("track_ids", []):
                    existing_tracks.append(track_id)
                    continue
                
                # Add track to collection
                if "track_ids" not in collection:
                    collection["track_ids"] = []
                collection["track_ids"].append(track_id)
                added_tracks.append(track_id)
            
            # Update timestamp
            if added_tracks:
                collection["updated_at"] = self._get_current_timestamp()
                self._save_collections()
            
            return {
                "status": "success",
                "message": f"Added {len(added_tracks)} tracks to collection '{collection.get('name')}'",
                "added_tracks": added_tracks,
                "existing_tracks": existing_tracks
            }
            
        except Exception as e:
            logger.error(f"Error adding tracks to collection: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def remove_tracks_from_collection(
        self,
        collection_id: str,
        track_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Remove tracks from a collection.
        
        Args:
            collection_id: ID of the collection
            track_ids: IDs of tracks to remove
            
        Returns:
            Dictionary with result of the operation
        """
        try:
            # Find collection
            collection = None
            for c in self.collections:
                if c.get("id") == collection_id:
                    collection = c
                    break
            
            if not collection:
                return {
                    "status": "error",
                    "error": f"Collection not found: {collection_id}"
                }
            
            # Remove tracks
            removed_tracks = []
            
            for track_id in track_ids:
                # Check if track is in collection
                if track_id in collection.get("track_ids", []):
                    collection["track_ids"].remove(track_id)
                    removed_tracks.append(track_id)
            
            # Update timestamp
            if removed_tracks:
                collection["updated_at"] = self._get_current_timestamp()
                self._save_collections()
            
            return {
                "status": "success",
                "message": f"Removed {len(removed_tracks)} tracks from collection '{collection.get('name')}'",
                "removed_tracks": removed_tracks
            }
            
        except Exception as e:
            logger.error(f"Error removing tracks from collection: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_collection(self, collection_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a collection by ID.
        
        Args:
            collection_id: ID of the collection
            
        Returns:
            Collection metadata or None if not found
        """
        for collection in self.collections:
            if collection.get("id") == collection_id:
                return collection
        return None
    
    def get_all_collections(self) -> List[Dict[str, Any]]:
        """
        Get all collections.
        
        Returns:
            List of collection metadata
        """
        return self.collections
    
    def get_collection_tracks(self, collection_id: str) -> List[Dict[str, Any]]:
        """
        Get tracks in a collection.
        
        Args:
            collection_id: ID of the collection
            
        Returns:
            List of track metadata
        """
        collection = self.get_collection(collection_id)
        if not collection:
            return []
        
        tracks = []
        for track_id in collection.get("track_ids", []):
            track = self.get_track(track_id)
            if track:
                tracks.append(track)
        
        return tracks
    
    def _load_metadata(self) -> List[Dict[str, Any]]:
        """Load track metadata from file."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {str(e)}")
        
        return []
    
    def _save_metadata(self) -> bool:
        """Save track metadata to file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.tracks, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
            return False
    
    def _load_collections(self) -> List[Dict[str, Any]]:
        """Load collections from file."""
        if os.path.exists(self.collections_file):
            try:
                with open(self.collections_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading collections: {str(e)}")
        
        return []
    
    def _save_collections(self) -> bool:
        """Save collections to file."""
        try:
            with open(self.collections_file, 'w') as f:
                json.dump(self.collections, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving collections: {str(e)}")
            return False
    
    def _get_audio_duration(self, file_path: str) -> float:
        """
        Get the duration of an audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Duration in seconds
        """
        try:
            import subprocess
            
            # Get ffprobe path from config or use default
            ffprobe_path = self.config.get('ffprobe_path', 'ffprobe')
            
            # Run ffprobe to get duration
            cmd = [
                ffprobe_path,
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                file_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse duration
            duration = float(result.stdout.strip())
            return duration
            
        except Exception as e:
            logger.error(f"Error getting audio duration: {str(e)}")
            return 0.0
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _calculate_relevance_score(
        self,
        track: Dict[str, Any],
        mood: Optional[str],
        tempo: Optional[float],
        genre: Optional[str],
        duration: Optional[float],
        keywords: Optional[List[str]]
    ) -> float:
        """
        Calculate relevance score for a track based on search criteria.
        
        Args:
            track: Track metadata
            mood: Target mood
            tempo: Target tempo
            genre: Target genre
            duration: Target duration
            keywords: Search keywords
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        score = 0.0
        factors = 0
        
        # Mood score
        if mood and "mood" in track:
            if track["mood"] == mood:
                score += 1.0
            else:
                # Check for similar moods
                similar_moods = self._get_similar_moods(mood)
                if track["mood"] in similar_moods:
                    score += 0.7
            factors += 1
        
        # Tempo score
        if tempo is not None and "bpm" in track:
            tempo_diff = abs(track["bpm"] - tempo)
            tempo_tolerance = self.config.get('tempo_tolerance', 10)
            tempo_score = max(0, 1.0 - (tempo_diff / tempo_tolerance))
            score += tempo_score
            factors += 1
        
        # Genre score
        if genre and "genre" in track:
            if track["genre"] == genre:
                score += 1.0
            else:
                # Check for compatible genres
                compatible_genres = self._get_compatible_genres(genre)
                if track["genre"] in compatible_genres:
                    score += 0.7
            factors += 1
        
        # Duration score
        if duration is not None and "duration" in track:
            duration_diff = abs(track["duration"] - duration)
            duration_tolerance = self.config.get('duration_tolerance', 30)
            duration_score = max(0, 1.0 - (duration_diff / duration_tolerance))
            score += duration_score
            factors += 1
        
        # Keyword score
        if keywords and len(keywords) > 0:
            keyword_matches = 0
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if (
                    keyword_lower in track.get("title", "").lower() or
                    keyword_lower in track.get("artist", "").lower() or
                    keyword_lower in track.get("description", "").lower() or
                    keyword_lower in " ".join(str(tag).lower() for tag in track.get("tags", []))
                ):
                    keyword_matches += 1
            
            keyword_score = keyword_matches / len(keywords)
            score += keyword_score
            factors += 1
        
        # Calculate average score
        if factors > 0:
            return score / factors
        else:
            return 0.0
    
    def _get_similar_moods(self, mood: str) -> List[str]:
        """
        Get similar moods based on valence-arousal model.
        
        Args:
            mood: Target mood
            
        Returns:
            List of similar moods
        """
        # Define mood groups based on similarity
        mood_groups = {
            "happy": ["joyful", "upbeat", "cheerful", "excited"],
            "sad": ["melancholic", "depressed", "gloomy", "somber"],
            "angry": ["aggressive", "tense", "frustrated", "intense"],
            "relaxed": ["calm", "peaceful", "serene", "tranquil"],
            "tender": ["romantic", "gentle", "soft", "warm"],
            "excited": ["energetic", "lively", "animated", "thrilling"],
            "nostalgic": ["reflective", "wistful", "reminiscent", "sentimental"],
            "suspenseful": ["mysterious", "tense", "dramatic", "eerie"],
            "inspiring": ["uplifting", "motivational", "empowering", "triumphant"]
        }
        
        # Find the group containing the mood
        for group_mood, similar_moods in mood_groups.items():
            if mood == group_mood or mood in similar_moods:
                # Return all moods in the group
                result = [group_mood] + similar_moods
                return [m for m in result if m != mood]
        
        # If mood not found in any group, return empty list
        return []
    
    def _get_compatible_genres(self, genre: str) -> List[str]:
        """
        Get compatible genres based on musical similarity.
        
        Args:
            genre: Target genre
            
        Returns:
            List of compatible genres
        """
        # Define genre compatibility groups
        genre_groups = {
            "rock": ["alternative", "indie", "pop_rock", "hard_rock", "classic_rock"],
            "pop": ["dance_pop", "electropop", "synth_pop", "indie_pop"],
            "electronic": ["edm", "techno", "house", "trance", "dubstep"],
            "hip_hop": ["rap", "trap", "r&b", "urban"],
            "jazz": ["blues", "soul", "funk", "fusion"],
            "classical": ["orchestral", "chamber", "piano", "opera"],
            "country": ["folk", "bluegrass", "americana"],
            "ambient": ["new_age", "chillout", "atmospheric"],
            "metal": ["heavy_metal", "thrash", "death_metal", "hard_rock"],
            "world": ["latin", "reggae", "afrobeat", "ethnic"]
        }
        
        # Find the group containing the genre
        for group_genre, compatible_genres in genre_groups.items():
            if genre == group_genre or genre in compatible_genres:
                # Return all genres in the group
                result = [group_genre] + compatible_genres
                return [g for g in result if g != genre]
        
        # If genre not found in any group, return empty list
        return [] 