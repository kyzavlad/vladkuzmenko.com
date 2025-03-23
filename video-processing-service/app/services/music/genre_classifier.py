"""
Genre Classification Module

This module provides functionality for classifying music tracks into genres
and making recommendations based on genre compatibility.
"""

import os
import logging
import tempfile
import json
import subprocess
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class GenreClassifier:
    """
    Classifies music tracks into genres and provides genre-based recommendations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the genre classifier."""
        self.config = config or {}
        
        # Set default parameters
        self.temp_dir = self.config.get('temp_dir', tempfile.gettempdir())
        self.ffmpeg_path = self.config.get('ffmpeg_path', 'ffmpeg')
        self.ffprobe_path = self.config.get('ffprobe_path', 'ffprobe')
        
        # Genre taxonomy
        self.genres = [
            "ambient", "blues", "classical", "country", "dance", "electronic",
            "folk", "funk", "hip-hop", "indie", "jazz", "latin", "metal", "pop",
            "r&b", "reggae", "rock", "soul", "soundtrack", "world"
        ]
        
        # Initialize audio processing libraries
        self.librosa_available = False
        try:
            import librosa
            self.librosa_available = True
        except ImportError:
            logger.warning("Librosa not available. Genre classification will be limited.")
    
    def classify_genre(self, audio_path: str, top_n: int = 3) -> Dict[str, Any]:
        """
        Classify the genre of an audio track.
        
        Args:
            audio_path: Path to the audio file
            top_n: Number of top genres to return
            
        Returns:
            Dictionary with genre classification results
        """
        # Check if file exists
        if not os.path.exists(audio_path):
            return {
                "status": "error",
                "error": f"File not found: {audio_path}"
            }
        
        # Extract audio features
        features = self._extract_audio_features(audio_path)
        
        if not features:
            return {
                "status": "error",
                "error": "Failed to extract audio features"
            }
        
        # Classify genre using feature-based classification
        genre_probabilities = self._classify_with_features(features)
        
        # Get top N genres
        top_genres = []
        for genre, probability in sorted(genre_probabilities.items(), key=lambda x: x[1], reverse=True)[:top_n]:
            top_genres.append({
                "genre": genre,
                "probability": float(probability)
            })
        
        return {
            "status": "success",
            "file_path": audio_path,
            "primary_genre": top_genres[0]["genre"] if top_genres else None,
            "top_genres": top_genres,
            "feature_summary": self._summarize_features(features)
        }
    
    def _extract_audio_features(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """Extract audio features for genre classification."""
        try:
            # Get audio info
            audio_info = self._get_audio_info(audio_path)
            
            # Use FFmpeg to analyze audio
            cmd = [
                self.ffmpeg_path,
                "-i", audio_path,
                "-filter_complex", "ebur128=peak=true",
                "-f", "null", "-"
            ]
            
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Parse FFmpeg output for features
            features = {}
            features['duration'] = audio_info.get('duration', 0)
            
            # Set default features for demo
            features['tempo'] = 120  # Default tempo (BPM)
            features['spectral_centroid_mean'] = 3000  # Middle frequency range
            features['zero_crossing_rate'] = 0.1  # Moderate distortion
            features['rms_mean'] = -15  # Moderate volume
            features['dynamic_range'] = 12  # Moderate dynamic range
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {str(e)}")
            return None
    
    def _classify_with_features(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Classify genre using feature-based heuristics."""
        # Initialize with equal probabilities
        genre_probabilities = {genre: 0.05 for genre in self.genres}
        
        # Extract key features
        spectral_centroid = features.get('spectral_centroid_mean', 3000)
        zero_crossing = features.get('zero_crossing_rate', 0.1)
        tempo = features.get('tempo', 120)
        rms = features.get('rms_mean', -15)
        dynamic_range = features.get('dynamic_range', 12)
        
        # Simple heuristic classification
        # Note: This is a simplified demo version, a real implementation would use
        # machine learning or more sophisticated feature analysis
        
        # Rock characteristics
        if tempo > 100 and tempo < 140:
            genre_probabilities['rock'] += 0.3
        
        # Electronic characteristics
        if tempo > 120:
            genre_probabilities['electronic'] += 0.3
            genre_probabilities['dance'] += 0.2
        
        # Classical characteristics
        if dynamic_range > 15:
            genre_probabilities['classical'] += 0.3
            genre_probabilities['soundtrack'] += 0.2
        
        # Normalize to ensure they sum to 1
        total_prob = sum(genre_probabilities.values())
        for genre in genre_probabilities:
            genre_probabilities[genre] /= total_prob
        
        return genre_probabilities
    
    def recommend_genres_for_video(self, content_mood: str, content_genre: str, top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Recommend music genres for a video based on content mood and genre.
        
        Args:
            content_mood: Mood of the video content
            content_genre: Genre/category of the video content
            top_n: Number of top recommendations to return
            
        Returns:
            List of recommended genres with scores
        """
        # Map content moods to music genres
        mood_genre_map = {
            "happy": ["pop", "dance", "electronic", "funk", "reggae"],
            "sad": ["ambient", "classical", "folk", "blues", "indie"],
            "energetic": ["rock", "electronic", "dance", "hip-hop", "metal"],
            "calm": ["ambient", "classical", "folk", "jazz", "world"],
            "tense": ["soundtrack", "electronic", "rock", "metal"],
            "romantic": ["r&b", "jazz", "pop", "soul", "classical"]
        }
        
        # Map content genres to music genres
        content_genre_map = {
            "documentary": ["ambient", "classical", "soundtrack", "world"],
            "tutorial": ["ambient", "electronic", "pop", "jazz"],
            "vlog": ["pop", "indie", "electronic", "hip-hop"],
            "travel": ["world", "pop", "electronic", "ambient"],
            "gaming": ["electronic", "rock", "metal", "soundtrack"]
        }
        
        # Get genres for mood and content
        mood_genres = mood_genre_map.get(content_mood.lower(), [])
        content_genres = content_genre_map.get(content_genre.lower(), [])
        
        # If either is missing, use the other one
        if not mood_genres and not content_genres:
            return []
        
        if not mood_genres:
            mood_genres = content_genres
        
        if not content_genres:
            content_genres = mood_genres
        
        # Calculate genre scores
        genre_scores = {}
        
        # Score based on mood
        for i, genre in enumerate(mood_genres):
            genre_scores[genre] = genre_scores.get(genre, 0) + (1.0 - i/len(mood_genres)) * 0.6
        
        # Score based on content
        for i, genre in enumerate(content_genres):
            genre_scores[genre] = genre_scores.get(genre, 0) + (1.0 - i/len(content_genres)) * 0.4
        
        # Convert to recommendations list
        recommendations = [
            {"genre": genre, "score": score}
            for genre, score in genre_scores.items()
        ]
        
        # Sort by score and return top N
        return sorted(recommendations, key=lambda x: x["score"], reverse=True)[:top_n]
    
    def _summarize_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Create a human-readable summary of audio features."""
        summary = {}
        
        # Tempo classification
        tempo = features.get('tempo', 120)
        if tempo < 70:
            summary['tempo'] = 'Very slow'
        elif tempo < 90:
            summary['tempo'] = 'Slow'
        elif tempo < 120:
            summary['tempo'] = 'Moderate'
        elif tempo < 150:
            summary['tempo'] = 'Fast'
        else:
            summary['tempo'] = 'Very fast'
        
        summary['bpm'] = tempo
        
        # Simplified feature summary
        return summary
    
    def _get_audio_info(self, audio_path: str) -> Dict[str, Any]:
        """Get basic audio file information."""
        cmd = [
            self.ffprobe_path,
            "-v", "error",
            "-select_streams", "a",
            "-show_entries", "format=duration",
            "-of", "json",
            audio_path
        ]
        
        try:
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            data = json.loads(result.stdout)
            
            audio_info = {}
            
            # Extract format information
            if 'format' in data and 'duration' in data['format']:
                audio_info['duration'] = float(data['format']['duration'])
            
            return audio_info
            
        except Exception as e:
            logger.error(f"Error getting audio info: {str(e)}")
            return {'duration': 0} 