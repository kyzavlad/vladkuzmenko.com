"""
Audio Fingerprinter Module

This module provides functionality for generating and comparing audio fingerprints,
which can be used to identify music tracks and prevent copyright issues.
"""

import os
import logging
import tempfile
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class AudioFingerprinter:
    """
    Generates and compares audio fingerprints.
    
    This class provides methods to create acoustic fingerprints of audio content,
    which can be used to identify music tracks and prevent copyright issues.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the audio fingerprinter.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        
        # Set default parameters
        self.temp_dir = self.config.get('temp_dir', tempfile.gettempdir())
        self.ffmpeg_path = self.config.get('ffmpeg_path', 'ffmpeg')
        self.ffprobe_path = self.config.get('ffprobe_path', 'ffprobe')
        self.fingerprint_db_path = self.config.get('fingerprint_db_path', os.path.join(os.getcwd(), 'fingerprint_db'))
        
        # Create fingerprint database directory if it doesn't exist
        os.makedirs(self.fingerprint_db_path, exist_ok=True)
        
        # Initialize audio processing libraries if available
        try:
            import librosa
            self.librosa_available = True
        except ImportError:
            logger.warning("Librosa not available. Audio fingerprinting will be limited.")
            self.librosa_available = False
    
    def generate_fingerprint(self, audio_path: str) -> Dict[str, Any]:
        """
        Generate an audio fingerprint for a file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary with fingerprint data
        """
        results = {
            "status": "success",
            "input_path": audio_path
        }
        
        # If librosa is not available, use a more basic method
        if not self.librosa_available:
            logger.warning("Librosa not available. Using basic fingerprinting.")
            basic_fingerprint = self._generate_basic_fingerprint(audio_path)
            results.update(basic_fingerprint)
            return results
        
        try:
            import librosa
            
            # Load audio file
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            
            # Extract features for fingerprinting
            
            # 1. Mel-frequency cepstral coefficients (MFCCs)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfcc_delta = librosa.feature.delta(mfccs)
            
            # 2. Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            
            # 3. Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            
            # 4. Spectral centroid
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            
            # Compute summary statistics for each feature
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_var = np.var(mfccs, axis=1)
            mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
            
            chroma_mean = np.mean(chroma, axis=1)
            chroma_var = np.var(chroma, axis=1)
            
            contrast_mean = np.mean(contrast, axis=1)
            centroid_mean = np.mean(centroid, axis=1)
            
            # Combine features into a fingerprint vector
            fingerprint_vector = np.concatenate([
                mfcc_mean, mfcc_var, mfcc_delta_mean,
                chroma_mean, chroma_var,
                contrast_mean, centroid_mean
            ])
            
            # Generate a hash of the fingerprint
            fingerprint_hash = hashlib.sha256(fingerprint_vector.tobytes()).hexdigest()
            
            # Create fingerprint data
            fingerprint_data = {
                "hash": fingerprint_hash,
                "vector": fingerprint_vector.tolist(),
                "features": {
                    "mfcc_mean": mfcc_mean.tolist(),
                    "mfcc_var": mfcc_var.tolist(),
                    "mfcc_delta_mean": mfcc_delta_mean.tolist(),
                    "chroma_mean": chroma_mean.tolist(),
                    "chroma_var": chroma_var.tolist(),
                    "contrast_mean": contrast_mean.tolist(),
                    "centroid_mean": centroid_mean.tolist()
                }
            }
            
            results["fingerprint"] = fingerprint_data
            
            return results
            
        except Exception as e:
            logger.error(f"Error generating fingerprint: {str(e)}")
            results["status"] = "error"
            results["error"] = str(e)
            
            # Try fallback method
            try:
                basic_fingerprint = self._generate_basic_fingerprint(audio_path)
                results.update(basic_fingerprint)
                results["status"] = "partial_success"
                results["note"] = "Used fallback method due to error in primary method"
            except Exception as fallback_error:
                results["fallback_error"] = str(fallback_error)
            
            return results
    
    def compare_fingerprints(
        self,
        fingerprint1: Dict[str, Any],
        fingerprint2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare two audio fingerprints.
        
        Args:
            fingerprint1: First fingerprint
            fingerprint2: Second fingerprint
            
        Returns:
            Dictionary with comparison results
        """
        results = {
            "status": "success"
        }
        
        try:
            # Check if fingerprints have vectors
            if "vector" not in fingerprint1 or "vector" not in fingerprint2:
                return {
                    "status": "error",
                    "error": "Fingerprints do not contain vector data"
                }
            
            # Convert vectors to numpy arrays
            vector1 = np.array(fingerprint1["vector"])
            vector2 = np.array(fingerprint2["vector"])
            
            # Calculate cosine similarity
            similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
            
            # Calculate Euclidean distance
            distance = np.linalg.norm(vector1 - vector2)
            
            # Determine if fingerprints match
            similarity_threshold = self.config.get('similarity_threshold', 0.95)
            distance_threshold = self.config.get('distance_threshold', 0.5)
            
            is_match = similarity >= similarity_threshold and distance <= distance_threshold
            
            results.update({
                "similarity": float(similarity),
                "distance": float(distance),
                "is_match": is_match
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Error comparing fingerprints: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def identify_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Identify an audio file by comparing its fingerprint to a database.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary with identification results
        """
        results = {
            "status": "success",
            "input_path": audio_path,
            "matches": []
        }
        
        try:
            # Generate fingerprint for the input audio
            fingerprint_result = self.generate_fingerprint(audio_path)
            
            if fingerprint_result["status"] != "success" and fingerprint_result["status"] != "partial_success":
                return {
                    "status": "error",
                    "error": f"Failed to generate fingerprint: {fingerprint_result.get('error', 'Unknown error')}"
                }
            
            input_fingerprint = fingerprint_result.get("fingerprint", {})
            
            # Load fingerprint database
            fingerprints = self._load_fingerprint_database()
            
            # Compare input fingerprint to database
            matches = []
            
            for db_fingerprint in fingerprints:
                comparison = self.compare_fingerprints(input_fingerprint, db_fingerprint["fingerprint"])
                
                if comparison.get("is_match", False):
                    matches.append({
                        "track_id": db_fingerprint.get("track_id"),
                        "title": db_fingerprint.get("title"),
                        "artist": db_fingerprint.get("artist"),
                        "similarity": comparison.get("similarity"),
                        "distance": comparison.get("distance")
                    })
            
            # Sort matches by similarity (highest first)
            matches.sort(key=lambda x: x.get("similarity", 0), reverse=True)
            
            results["matches"] = matches
            results["match_count"] = len(matches)
            
            if matches:
                results["best_match"] = matches[0]
            
            return results
            
        except Exception as e:
            logger.error(f"Error identifying audio: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def add_to_database(
        self,
        audio_path: str,
        track_id: str,
        title: str,
        artist: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add an audio fingerprint to the database.
        
        Args:
            audio_path: Path to the audio file
            track_id: ID of the track
            title: Title of the track
            artist: Artist of the track
            metadata: Additional metadata
            
        Returns:
            Dictionary with result of the operation
        """
        try:
            # Generate fingerprint
            fingerprint_result = self.generate_fingerprint(audio_path)
            
            if fingerprint_result["status"] != "success" and fingerprint_result["status"] != "partial_success":
                return {
                    "status": "error",
                    "error": f"Failed to generate fingerprint: {fingerprint_result.get('error', 'Unknown error')}"
                }
            
            fingerprint = fingerprint_result.get("fingerprint", {})
            
            # Create database entry
            entry = {
                "track_id": track_id,
                "title": title,
                "artist": artist,
                "fingerprint": fingerprint,
                "added_at": self._get_current_timestamp()
            }
            
            # Add additional metadata if provided
            if metadata:
                entry["metadata"] = metadata
            
            # Load existing database
            database = self._load_fingerprint_database()
            
            # Check if track already exists
            for i, db_entry in enumerate(database):
                if db_entry.get("track_id") == track_id:
                    # Update existing entry
                    database[i] = entry
                    self._save_fingerprint_database(database)
                    return {
                        "status": "success",
                        "message": f"Updated fingerprint for track '{title}'"
                    }
            
            # Add new entry
            database.append(entry)
            
            # Save database
            self._save_fingerprint_database(database)
            
            return {
                "status": "success",
                "message": f"Added fingerprint for track '{title}'"
            }
            
        except Exception as e:
            logger.error(f"Error adding to database: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def remove_from_database(self, track_id: str) -> Dict[str, Any]:
        """
        Remove a track from the fingerprint database.
        
        Args:
            track_id: ID of the track
            
        Returns:
            Dictionary with result of the operation
        """
        try:
            # Load database
            database = self._load_fingerprint_database()
            
            # Find track
            for entry in database:
                if entry.get("track_id") == track_id:
                    # Remove entry
                    database = [e for e in database if e.get("track_id") != track_id]
                    
                    # Save database
                    self._save_fingerprint_database(database)
                    
                    return {
                        "status": "success",
                        "message": f"Removed fingerprint for track ID '{track_id}'"
                    }
            
            return {
                "status": "error",
                "error": f"Track with ID '{track_id}' not found in database"
            }
            
        except Exception as e:
            logger.error(f"Error removing from database: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _generate_basic_fingerprint(self, audio_path: str) -> Dict[str, Any]:
        """
        Generate a basic audio fingerprint using FFmpeg.
        
        This is a fallback method when librosa is not available.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary with basic fingerprint data
        """
        import subprocess
        import hashlib
        
        # Create a temporary file for the audio data
        with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Extract raw audio data
            cmd = [
                self.ffmpeg_path,
                "-i", audio_path,
                "-ac", "1",  # Mono
                "-ar", "22050",  # Sample rate
                "-f", "s16le",  # 16-bit signed little-endian
                "-y",  # Overwrite output file
                temp_path
            ]
            
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Read raw audio data
            with open(temp_path, "rb") as f:
                audio_data = f.read()
            
            # Calculate hash of audio data
            audio_hash = hashlib.sha256(audio_data).hexdigest()
            
            # Create a simple fingerprint
            fingerprint = {
                "hash": audio_hash,
                "method": "basic",
                "vector": []  # Empty vector for compatibility
            }
            
            return {
                "fingerprint": fingerprint
            }
            
        except Exception as e:
            logger.error(f"Error generating basic fingerprint: {str(e)}")
            return {
                "error": str(e)
            }
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except Exception:
                pass
    
    def _load_fingerprint_database(self) -> List[Dict[str, Any]]:
        """
        Load the fingerprint database.
        
        Returns:
            List of fingerprint entries
        """
        db_file = os.path.join(self.fingerprint_db_path, "fingerprints.json")
        
        if os.path.exists(db_file):
            try:
                with open(db_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading fingerprint database: {str(e)}")
        
        return []
    
    def _save_fingerprint_database(self, database: List[Dict[str, Any]]) -> bool:
        """
        Save the fingerprint database.
        
        Args:
            database: List of fingerprint entries
            
        Returns:
            True if successful, False otherwise
        """
        db_file = os.path.join(self.fingerprint_db_path, "fingerprints.json")
        
        try:
            with open(db_file, "w") as f:
                json.dump(database, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving fingerprint database: {str(e)}")
            return False
    
    def _get_current_timestamp(self) -> str:
        """
        Get current timestamp in ISO format.
        
        Returns:
            Current timestamp
        """
        from datetime import datetime
        return datetime.now().isoformat() 