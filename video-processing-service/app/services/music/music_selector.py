"""
Music Selector Module

This module provides functionality for selecting appropriate music tracks
for video content based on mood, tempo, genre, and other factors.
"""

import os
import logging
import tempfile
import json
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class MusicSelector:
    """
    Selects appropriate music tracks for video content.
    
    This class provides methods to analyze video content and select music
    tracks that match the mood, tempo, and other characteristics of the video.
    It integrates various components such as mood analysis, BPM detection,
    genre classification, and audio fingerprinting for a comprehensive
    music selection experience.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the music selector.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        
        # Set default parameters
        self.temp_dir = self.config.get('temp_dir', tempfile.gettempdir())
        self.ffmpeg_path = self.config.get('ffmpeg_path', 'ffmpeg')
        self.ffprobe_path = self.config.get('ffprobe_path', 'ffprobe')
        
        # Initialize components
        from app.services.music.mood_analyzer import MoodAnalyzer
        from app.services.music.bpm_detector import BPMDetector
        from app.services.music.genre_classifier import GenreClassifier
        from app.services.music.music_library import MusicLibrary
        from app.services.music.audio_fingerprinter import AudioFingerprinter
        from app.services.music.emotional_arc_mapper import EmotionalArcMapper
        
        self.mood_analyzer = MoodAnalyzer(config)
        self.bpm_detector = BPMDetector(config)
        self.genre_classifier = GenreClassifier(config) if self.config.get('use_genre_classifier', True) else None
        self.music_library = MusicLibrary(config)
        self.audio_fingerprinter = AudioFingerprinter(config) if self.config.get('use_audio_fingerprinter', True) else None
        self.emotional_arc_mapper = EmotionalArcMapper(config) if self.config.get('use_emotional_arc_mapper', True) else None
    
    def select_music(
        self,
        video_path: str,
        transcript: Optional[List[Dict[str, Any]]] = None,
        target_duration: Optional[float] = None,
        mood_override: Optional[str] = None,
        tempo_override: Optional[float] = None,
        genre_override: Optional[str] = None,
        max_results: int = 5,
        copyright_free_only: bool = False,
        collection_id: Optional[str] = None,
        use_emotional_arc: bool = False,
        segment_duration: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Select appropriate music tracks for a video.
        
        Args:
            video_path: Path to the video file
            transcript: Optional transcript data (list of segments)
            target_duration: Optional target duration for the music
            mood_override: Optional mood override
            tempo_override: Optional tempo override
            genre_override: Optional genre override
            max_results: Maximum number of results to return
            copyright_free_only: Only return copyright-free tracks
            collection_id: ID of collection to search in
            use_emotional_arc: Whether to analyze emotional arc for better music selection
            segment_duration: Duration of segments for timeline analysis (seconds)
            
        Returns:
            Dictionary with music selection results
        """
        results = {
            "status": "success",
            "input_path": video_path,
            "recommendations": []
        }
        
        try:
            # Get video duration if target_duration is not specified
            if target_duration is None:
                target_duration = self._get_video_duration(video_path)
                results["video_duration"] = target_duration
            
            # Analyze video mood if mood_override is not specified
            mood_results = None
            if mood_override is None:
                mood_results = self.mood_analyzer.analyze_mood(
                    video_path=video_path,
                    transcript=transcript
                )
                results["mood_analysis"] = mood_results
                
                # Extract mood labels
                if "overall_mood" in mood_results and "mood_labels" in mood_results["overall_mood"]:
                    mood_labels = mood_results["overall_mood"]["mood_labels"]
                else:
                    mood_labels = []
            else:
                mood_labels = [mood_override]
            
            # Add emotional arc analysis if requested
            emotional_arc_results = None
            if use_emotional_arc and self.emotional_arc_mapper is not None:
                emotional_arc_results = self.emotional_arc_mapper.map_emotional_arc(
                    video_path=video_path,
                    transcript=transcript,
                    segment_duration=segment_duration or 5,
                    detect_key_moments=True,
                    smooth_arc=True,
                    use_existing_mood_analysis=mood_results is not None
                )
                results["emotional_arc"] = emotional_arc_results
            
            # Detect video BPM if tempo_override is not specified
            target_bpm = tempo_override
            if target_bpm is None and mood_results is not None:
                # Try to infer BPM from mood
                target_bpm = self._estimate_bpm_from_mood(mood_results)
            
            # Detect video genre if genre_override is not specified
            target_genre = genre_override
            if target_genre is None and mood_results is not None:
                # Try to infer genre from mood
                target_genre = self._estimate_genre_from_mood(mood_results)
            
            # Search music library for matching tracks
            search_params = {
                "max_results": max_results * 2,  # Get more results than needed for post-processing
                "copyright_free_only": copyright_free_only,
                "collection_id": collection_id
            }
            
            # Add mood to search parameters
            if mood_labels:
                search_params["mood"] = mood_labels[0]  # Use primary mood
            
            # Add tempo to search parameters
            if target_bpm is not None:
                search_params["tempo"] = target_bpm
            
            # Add genre to search parameters
            if target_genre is not None:
                search_params["genre"] = target_genre
            
            # Add duration to search parameters
            if target_duration is not None:
                search_params["duration"] = target_duration
            
            # Search for tracks
            search_results = self.music_library.search_tracks(**search_params)
            
            # If no results found, try with looser criteria
            if not search_results.get("tracks", []):
                # Try with just mood
                if "mood" in search_params:
                    looser_params = {"mood": search_params["mood"], "max_results": max_results * 2}
                    search_results = self.music_library.search_tracks(**looser_params)
                
                # If still no results, try with just genre
                if not search_results.get("tracks", []) and "genre" in search_params:
                    looser_params = {"genre": search_params["genre"], "max_results": max_results * 2}
                    search_results = self.music_library.search_tracks(**looser_params)
                
                # If still no results, try with just tempo
                if not search_results.get("tracks", []) and "tempo" in search_params:
                    looser_params = {"tempo": search_params["tempo"], "max_results": max_results * 2}
                    search_results = self.music_library.search_tracks(**looser_params)
                
                # If still no results, get some random tracks
                if not search_results.get("tracks", []):
                    looser_params = {"max_results": max_results * 2}
                    if copyright_free_only:
                        looser_params["copyright_free_only"] = True
                    search_results = self.music_library.search_tracks(**looser_params)
            
            # Post-process results for more accurate ranking
            tracks = search_results.get("tracks", [])
            
            if tracks:
                # Check for copyright issues if fingerprinter is available
                if self.audio_fingerprinter is not None and not copyright_free_only:
                    tracks = self._filter_copyright_issues(tracks, video_path)
                
                # Re-rank tracks based on comprehensive scoring
                scored_tracks = self._score_tracks(
                    tracks=tracks,
                    target_mood=mood_labels[0] if mood_labels else None,
                    target_bpm=target_bpm,
                    target_genre=target_genre,
                    target_duration=target_duration,
                    emotional_arc=emotional_arc_results
                )
                
                # Sort by score (descending) and limit to max_results
                scored_tracks.sort(key=lambda x: x[1], reverse=True)
                top_tracks = [track for track, score in scored_tracks[:max_results]]
                
                # Add recommendations to results
                results["recommendations"] = top_tracks
                results["total_matches"] = len(tracks)
            else:
                results["recommendations"] = []
                results["total_matches"] = 0
                results["note"] = "No matching tracks found"
            
            return results
            
        except Exception as e:
            logger.error(f"Error selecting music: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def select_music_for_emotional_arc(
        self,
        video_path: str,
        transcript: Optional[List[Dict[str, Any]]] = None,
        segment_duration: int = 5,
        copyright_free_only: bool = False,
        collection_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Select appropriate music tracks based on the emotional arc of the video.
        
        This method analyzes the emotional arc of the video and selects music
        tracks that match the emotional journey, including key transitions and moments.
        
        Args:
            video_path: Path to the video file
            transcript: Optional transcript data (list of segments)
            segment_duration: Duration of segments for timeline analysis (seconds)
            copyright_free_only: Only return copyright-free tracks
            collection_id: ID of collection to search in
            
        Returns:
            Dictionary with music selection results
        """
        results = {
            "status": "success",
            "input_path": video_path,
            "recommendations": []
        }
        
        try:
            # Ensure emotional arc mapper is available
            if self.emotional_arc_mapper is None:
                return {
                    "status": "error",
                    "error": "Emotional arc mapper is not available"
                }
            
            # Map emotional arc
            arc_results = self.emotional_arc_mapper.map_emotional_arc(
                video_path=video_path,
                transcript=transcript,
                segment_duration=segment_duration,
                detect_key_moments=True,
                smooth_arc=True
            )
            
            results["emotional_arc"] = arc_results
            
            # Extract key moments and cue points
            key_moments = arc_results.get("key_moments", [])
            music_cues = arc_results.get("music_cues", [])
            
            # Create a timeline of music segments
            music_timeline = self._create_music_timeline(
                video_path=video_path,
                emotional_arc=arc_results,
                music_cues=music_cues
            )
            
            results["music_timeline"] = music_timeline
            
            # Find tracks for each segment in the timeline
            segment_tracks = []
            
            for segment in music_timeline:
                # Search for tracks matching the segment requirements
                segment_mood = segment.get("mood")
                segment_duration = segment.get("duration", 30)
                
                search_params = {
                    "mood": segment_mood,
                    "duration": segment_duration,
                    "max_results": 3,
                    "copyright_free_only": copyright_free_only,
                    "collection_id": collection_id
                }
                
                # Search for tracks
                search_results = self.music_library.search_tracks(**search_params)
                
                if search_results.get("tracks", []):
                    # Get the top track for this segment
                    top_track = search_results["tracks"][0]
                    
                    # Add segment information to track
                    top_track["segment"] = segment
                    
                    segment_tracks.append(top_track)
                else:
                    # No tracks found for this segment
                    segment_tracks.append({
                        "segment": segment,
                        "note": f"No tracks found for {segment_mood} mood"
                    })
            
            results["segment_tracks"] = segment_tracks
            
            # Find tracks for the entire video (fallback)
            overall_results = self.select_music(
                video_path=video_path,
                transcript=transcript,
                max_results=3,
                copyright_free_only=copyright_free_only,
                collection_id=collection_id,
                use_emotional_arc=True,
                segment_duration=segment_duration
            )
            
            results["overall_recommendations"] = overall_results.get("recommendations", [])
            
            return results
            
        except Exception as e:
            logger.error(f"Error selecting music for emotional arc: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def check_copyright(self, audio_path: str) -> Dict[str, Any]:
        """
        Check if an audio file has potential copyright issues.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary with copyright check results
        """
        results = {
            "status": "success",
            "input_path": audio_path,
            "potential_copyright_issues": False
        }
        
        try:
            # Ensure audio fingerprinter is available
            if self.audio_fingerprinter is None:
                return {
                    "status": "error",
                    "error": "Audio fingerprinter is not available"
                }
            
            # Identify audio
            identification = self.audio_fingerprinter.identify_audio(audio_path)
            
            if identification["status"] != "success":
                results["error"] = f"Error identifying audio: {identification.get('error', 'Unknown error')}"
                results["status"] = "error"
                return results
            
            # Check for matches
            matches = identification.get("matches", [])
            
            if matches:
                results["potential_copyright_issues"] = True
                results["matches"] = matches
            
            return results
            
        except Exception as e:
            logger.error(f"Error checking copyright: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _get_video_duration(self, video_path: str) -> float:
        """
        Get the duration of a video in seconds.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Duration in seconds
        """
        try:
            import subprocess
            import json
            
            cmd = [
                self.ffprobe_path,
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "format=duration",
                "-of", "json",
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            data = json.loads(result.stdout)
            
            return float(data["format"]["duration"])
            
        except Exception as e:
            logger.error(f"Error getting video duration: {str(e)}")
            return 0.0
    
    def _estimate_bpm_from_mood(self, mood_results: Dict[str, Any]) -> Optional[float]:
        """
        Estimate a target BPM based on mood analysis.
        
        Args:
            mood_results: Mood analysis results
            
        Returns:
            Estimated BPM or None if not possible
        """
        try:
            # Extract mood and arousal
            arousal = 0.5  # default middle value
            
            if "valence_arousal" in mood_results:
                arousal = mood_results["valence_arousal"].get("arousal", 0.5)
            
            # Map arousal to BPM range
            # Low arousal (0.0) -> Slow BPM (60)
            # High arousal (1.0) -> Fast BPM (160)
            bpm = 60 + (arousal * 100)
            
            return bpm
            
        except Exception as e:
            logger.error(f"Error estimating BPM from mood: {str(e)}")
            return None
    
    def _estimate_genre_from_mood(self, mood_results: Dict[str, Any]) -> Optional[str]:
        """
        Estimate a target genre based on mood analysis.
        
        Args:
            mood_results: Mood analysis results
            
        Returns:
            Estimated genre or None if not possible
        """
        try:
            # Extract mood labels
            if "overall_mood" in mood_results and "mood_labels" in mood_results["overall_mood"]:
                mood_labels = mood_results["overall_mood"]["mood_labels"]
                primary_mood = mood_labels[0] if mood_labels else None
            else:
                return None
            
            # Simple mood to genre mapping
            mood_genre_map = {
                "happy": "pop",
                "exciting": "rock",
                "relaxed": "ambient",
                "sad": "classical",
                "tense": "electronic",
                "scary": "electronic",
                "inspiring": "orchestral",
                "romantic": "jazz",
                "nostalgic": "jazz",
                "mysterious": "electronic",
                "angry": "rock",
                "energetic": "electronic",
                "calm": "ambient",
                "dreamy": "ambient",
                "playful": "pop"
            }
            
            return mood_genre_map.get(primary_mood)
            
        except Exception as e:
            logger.error(f"Error estimating genre from mood: {str(e)}")
            return None
    
    def _filter_copyright_issues(self, tracks: List[Dict[str, Any]], video_path: str) -> List[Dict[str, Any]]:
        """
        Filter out tracks that might have copyright issues.
        
        Args:
            tracks: List of tracks to filter
            video_path: Path to the video file
            
        Returns:
            Filtered list of tracks
        """
        if not self.audio_fingerprinter:
            return tracks
        
        try:
            # Check if any tracks are already copyright free
            copyright_free_tracks = [t for t in tracks if t.get("copyright_free", False)]
            
            # If we have copyright free tracks, prefer those
            if copyright_free_tracks:
                return copyright_free_tracks
            
            # Otherwise, check all tracks for potential copyright issues
            safe_tracks = []
            
            for track in tracks:
                # Skip if track has no file path
                if "file_path" not in track:
                    continue
                
                # Check if track exists in fingerprint database
                track_path = track["file_path"]
                
                identification = self.audio_fingerprinter.identify_audio(track_path)
                
                # If no matches found or identification failed, consider it safe
                if identification["status"] != "success" or not identification.get("matches", []):
                    safe_tracks.append(track)
            
            return safe_tracks if safe_tracks else tracks
            
        except Exception as e:
            logger.error(f"Error filtering copyright issues: {str(e)}")
            return tracks
    
    def _score_tracks(
        self,
        tracks: List[Dict[str, Any]],
        target_mood: Optional[str],
        target_bpm: Optional[float],
        target_genre: Optional[str],
        target_duration: Optional[float],
        emotional_arc: Optional[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Score tracks based on how well they match the target criteria.
        
        Args:
            tracks: List of tracks to score
            target_mood: Target mood
            target_bpm: Target BPM
            target_genre: Target genre
            target_duration: Target duration
            emotional_arc: Emotional arc analysis results
            
        Returns:
            List of (track, score) tuples
        """
        scored_tracks = []
        
        for track in tracks:
            score = 0.0
            factors = 0
            
            # Use existing relevance score if available
            if "relevance_score" in track:
                score += track["relevance_score"]
                factors += 1
            
            # Score based on mood
            if target_mood and "mood" in track:
                if track["mood"] == target_mood:
                    score += 1.0
                else:
                    # Check for similar moods
                    similar_moods = self._get_similar_moods(target_mood)
                    if track["mood"] in similar_moods:
                        score += 0.7
                factors += 1
            
            # Score based on BPM
            if target_bpm is not None and "bpm" in track:
                bpm_diff = abs(track["bpm"] - target_bpm)
                # Higher score for closer BPM match
                bpm_score = max(0, 1.0 - (bpm_diff / 30.0))  # Tolerance of 30 BPM
                score += bpm_score
                factors += 1
            
            # Score based on genre
            if target_genre and "genre" in track:
                if track["genre"] == target_genre:
                    score += 1.0
                else:
                    # Check for compatible genres
                    compatible_genres = self._get_compatible_genres(target_genre)
                    if track["genre"] in compatible_genres:
                        score += 0.7
                factors += 1
            
            # Score based on duration
            if target_duration is not None and "duration" in track:
                duration_diff = abs(track["duration"] - target_duration)
                # Higher score for closer duration match
                duration_score = max(0, 1.0 - (duration_diff / 60.0))  # Tolerance of 60 seconds
                score += duration_score
                factors += 1
            
            # Score based on emotional arc (if available)
            if emotional_arc is not None:
                # TODO: Implement scoring based on emotional arc
                pass
            
            # Bonus for copyright free tracks
            if track.get("copyright_free", False):
                score += 0.5
            
            # Calculate average score
            if factors > 0:
                final_score = score / factors
            else:
                final_score = 0.0
            
            scored_tracks.append((track, final_score))
        
        return scored_tracks
    
    def _create_music_timeline(
        self,
        video_path: str,
        emotional_arc: Dict[str, Any],
        music_cues: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Create a timeline of music segments based on emotional arc and cue points.
        
        Args:
            video_path: Path to the video file
            emotional_arc: Emotional arc analysis results
            music_cues: Music cue points from emotional arc analysis
            
        Returns:
            List of music segments
        """
        # Get video duration
        video_duration = self._get_video_duration(video_path)
        
        # Create segments based on music cues
        segments = []
        
        for i in range(len(music_cues)):
            start_time = music_cues[i]["time"]
            
            # Calculate end time (next cue point or end of video)
            if i < len(music_cues) - 1:
                end_time = music_cues[i + 1]["time"]
            else:
                end_time = video_duration
            
            # Create segment
            segment = {
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "mood": music_cues[i]["mood"],
                "cue_type": music_cues[i]["cue_type"],
                "description": music_cues[i].get("description", "")
            }
            
            segments.append(segment)
        
        return segments
    
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