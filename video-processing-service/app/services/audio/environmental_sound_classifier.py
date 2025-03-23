"""
Environmental Sound Classifier Module

This module provides functionality for classifying environmental sounds in audio recordings,
which is useful for identifying background noise types and improving noise reduction strategies.
"""

import os
import numpy as np
import logging
import tempfile
import librosa
import soundfile as sf
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pickle
import hashlib
import warnings

logger = logging.getLogger(__name__)

class EnvironmentalSoundClassifier:
    """
    A class for classifying environmental sounds in audio recordings.
    
    This class uses audio feature extraction and pre-trained models to identify
    common environmental sounds and background noises in audio recordings.
    """
    
    # Common environmental sound classes
    SOUND_CLASSES = [
        "air_conditioner", "car_horn", "children_playing", "dog_bark",
        "drilling", "engine_idling", "gun_shot", "jackhammer", "siren",
        "street_music", "speech", "crowd", "typing", "footsteps", "rain",
        "wind", "restaurant_chatter", "office_noise", "traffic", "white_noise"
    ]
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        feature_type: str = "mfcc",
        cache_dir: Optional[str] = None,
        confidence_threshold: float = 0.5,
        segment_length: float = 1.0,
        hop_length: float = 0.5
    ):
        """
        Initialize the EnvironmentalSoundClassifier with customizable parameters.
        
        Args:
            model_path: Path to a pre-trained model file (pickle format)
            feature_type: Type of features to extract ('mfcc', 'mel', or 'combined')
            cache_dir: Directory to cache extracted features and classification results
            confidence_threshold: Minimum confidence level for classification
            segment_length: Length of audio segments to analyze (in seconds)
            hop_length: Hop length between segments (in seconds)
        """
        self.model_path = model_path
        self.feature_type = feature_type
        self.cache_dir = cache_dir
        self.confidence_threshold = confidence_threshold
        self.segment_length = segment_length
        self.hop_length = hop_length
        
        # Initialize the model if provided
        self.model = None
        if self.model_path and os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"Loaded environmental sound classifier model from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load model from {self.model_path}: {str(e)}")
        
        # Create cache directory if specified
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def classify_audio(
        self,
        audio_path: str,
        return_all_segments: bool = False,
        return_features: bool = False
    ) -> Dict[str, Any]:
        """
        Classify environmental sounds in an audio file.
        
        Args:
            audio_path: Path to the input audio file
            return_all_segments: Whether to return classifications for all segments
            return_features: Whether to include extracted features in results
            
        Returns:
            Dictionary with classification results
        """
        try:
            # Check if we have cached results
            cache_id = None
            if self.cache_dir:
                cache_id = self._get_cache_id(audio_path)
                cache_file = os.path.join(self.cache_dir, f"{cache_id}.pkl")
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'rb') as f:
                            cached_results = pickle.load(f)
                        logger.info(f"Using cached classification results for {audio_path}")
                        
                        # Filter results based on parameters
                        if not return_all_segments and 'segment_classifications' in cached_results:
                            del cached_results['segment_classifications']
                        
                        if not return_features and 'features' in cached_results:
                            del cached_results['features']
                            
                        return cached_results
                    except Exception as e:
                        logger.warning(f"Failed to load cached results: {str(e)}")
            
            # Load the audio file
            y, sr = librosa.load(audio_path, sr=22050, mono=True)
            
            # Warn if no model is loaded
            if self.model is None:
                logger.warning("No classification model loaded, using feature-based heuristics")
            
            # Extract features from audio segments
            segment_samples = int(self.segment_length * sr)
            hop_samples = int(self.hop_length * sr)
            
            segment_times = []
            segment_features = []
            
            # Iterate through audio segments
            for start_sample in range(0, len(y) - segment_samples + 1, hop_samples):
                segment = y[start_sample:start_sample + segment_samples]
                
                # Calculate segment time in seconds
                start_time = start_sample / sr
                end_time = (start_sample + segment_samples) / sr
                segment_times.append((start_time, end_time))
                
                # Extract features
                features = self._extract_features(segment, sr)
                segment_features.append(features)
            
            # No segments were processed (audio too short)
            if not segment_features:
                return {
                    "status": "warning",
                    "message": "Audio file too short for classification",
                    "input_path": audio_path,
                    "dominant_sounds": []
                }
            
            # Classify each segment
            segment_classifications = []
            all_detections = {}
            
            for i, features in enumerate(segment_features):
                # If we have a model, use it for classification
                if self.model is not None:
                    # Reshape features for model input
                    features_flat = features.flatten().reshape(1, -1)
                    
                    # Get predictions
                    try:
                        # For sklearn models
                        if hasattr(self.model, 'predict_proba'):
                            predictions = self.model.predict_proba(features_flat)[0]
                            class_idx = np.argmax(predictions)
                            confidence = predictions[class_idx]
                            sound_class = self.SOUND_CLASSES[class_idx]
                        else:
                            # Basic prediction
                            class_idx = self.model.predict(features_flat)[0]
                            confidence = 0.8  # Default confidence
                            sound_class = self.SOUND_CLASSES[class_idx]
                    except Exception as e:
                        logger.error(f"Model prediction failed: {str(e)}")
                        # Fall back to feature-based classification
                        sound_class, confidence = self._classify_by_features(features)
                else:
                    # No model, use feature-based classification
                    sound_class, confidence = self._classify_by_features(features)
                
                # Add to segment classifications
                segment_classifications.append({
                    "start_time": segment_times[i][0],
                    "end_time": segment_times[i][1],
                    "class": sound_class,
                    "confidence": float(confidence)
                })
                
                # Count occurrences for overall classification
                if confidence >= self.confidence_threshold:
                    if sound_class not in all_detections:
                        all_detections[sound_class] = []
                    all_detections[sound_class].append(confidence)
            
            # Calculate overall sound presence
            dominant_sounds = []
            for sound_class, confidences in all_detections.items():
                # Calculate mean confidence and percentage of segments
                mean_confidence = np.mean(confidences)
                segment_percentage = len(confidences) / len(segment_classifications)
                
                dominant_sounds.append({
                    "class": sound_class,
                    "mean_confidence": float(mean_confidence),
                    "segment_percentage": float(segment_percentage),
                    "overall_score": float(mean_confidence * segment_percentage)
                })
            
            # Sort by overall score
            dominant_sounds.sort(key=lambda x: x["overall_score"], reverse=True)
            
            # Prepare result
            result = {
                "status": "success",
                "input_path": audio_path,
                "dominant_sounds": dominant_sounds,
                "processed_segments": len(segment_classifications)
            }
            
            # Include segment classifications if requested
            if return_all_segments:
                result["segment_classifications"] = segment_classifications
            
            # Include features if requested
            if return_features:
                result["features"] = segment_features
            
            # Cache results if cache directory specified
            if self.cache_dir and cache_id:
                try:
                    cache_file = os.path.join(self.cache_dir, f"{cache_id}.pkl")
                    with open(cache_file, 'wb') as f:
                        pickle.dump(result, f)
                    logger.debug(f"Cached classification results for {audio_path}")
                except Exception as e:
                    logger.warning(f"Failed to cache results: {str(e)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error classifying environmental sounds: {str(e)}")
            return {
                "status": "error",
                "input_path": audio_path,
                "error": str(e)
            }
    
    def _extract_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract audio features for classification.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Feature matrix
        """
        # Ensure non-empty audio segment
        if len(y) == 0:
            return np.zeros((128, 20))  # Default empty feature size
        
        # Handle feature extraction errors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if self.feature_type == "mfcc" or self.feature_type == "combined":
                # Extract MFCCs
                try:
                    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                    mfcc_delta = librosa.feature.delta(mfccs)
                    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
                    mfcc_features = np.vstack([mfccs, mfcc_delta, mfcc_delta2])
                except:
                    # Fall back to zeros if extraction fails
                    mfcc_features = np.zeros((39, 20))
            
            if self.feature_type == "mel" or self.feature_type == "combined":
                # Extract mel spectrogram
                try:
                    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
                except:
                    # Fall back to zeros if extraction fails
                    log_mel_spec = np.zeros((128, 20))
            
            # Return based on feature type
            if self.feature_type == "mfcc":
                return mfcc_features
            elif self.feature_type == "mel":
                return log_mel_spec
            else:  # combined
                # Resize features to match dimensions if needed
                if mfcc_features.shape[1] != log_mel_spec.shape[1]:
                    target_width = min(mfcc_features.shape[1], log_mel_spec.shape[1])
                    mfcc_features = mfcc_features[:, :target_width]
                    log_mel_spec = log_mel_spec[:, :target_width]
                return np.vstack([mfcc_features, log_mel_spec])
    
    def _classify_by_features(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Simple feature-based classification when no model is available.
        
        Args:
            features: Audio features
            
        Returns:
            Tuple of (detected_class, confidence)
        """
        # This is a very simplified approach 
        # In a real implementation, this would be more sophisticated
        
        # If features are MFCC
        if features.shape[0] < 100:
            # Analyze energy distribution in MFCCs
            mfcc_mean = np.mean(features, axis=1)
            
            # Simplified rules based on MFCC patterns
            if np.std(features) < 0.5:
                return "white_noise", 0.7
            elif np.abs(mfcc_mean[1]) > 2 * np.abs(mfcc_mean[2]):
                return "speech", 0.6
            elif np.abs(mfcc_mean[3]) > 2 * np.abs(mfcc_mean[1]):
                return "traffic", 0.5
            else:
                return "office_noise", 0.4
        
        # If features are mel spectrogram or combined
        else:
            # Analyze frequency distribution
            freq_energy = np.mean(features, axis=1)
            
            # Check for patterns in the frequency distribution
            low_energy = np.mean(freq_energy[:40])
            mid_energy = np.mean(freq_energy[40:80])
            high_energy = np.mean(freq_energy[80:])
            
            if high_energy > mid_energy and high_energy > low_energy:
                return "siren", 0.6
            elif low_energy > mid_energy and low_energy > high_energy:
                return "engine_idling", 0.5
            elif mid_energy > low_energy and mid_energy > high_energy:
                return "speech", 0.7
            else:
                return "restaurant_chatter", 0.4
    
    def _get_cache_id(self, file_path: str) -> str:
        """
        Generate a cache ID for a file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Cache ID string
        """
        # Get file stats
        stats = os.stat(file_path)
        
        # Create a unique identifier based on file path, size, and modification time
        unique_data = f"{file_path}_{stats.st_size}_{stats.st_mtime}"
        return hashlib.md5(unique_data.encode()).hexdigest()
    
    def get_noise_profile_for_class(self, sound_class: str) -> Dict[str, Any]:
        """
        Get recommended noise reduction profile for a specific environmental sound class.
        
        Args:
            sound_class: The environmental sound class
            
        Returns:
            Dictionary with noise reduction parameters
        """
        # Define noise profiles for different environmental sounds
        profiles = {
            "air_conditioner": {
                "description": "Steady, low-frequency noise",
                "noise_reduction": {
                    "reduction_strength": 0.7,
                    "freq_mask_smooth_hz": 600,
                    "time_mask_smooth_ms": 100
                }
            },
            "traffic": {
                "description": "Variable low-mid frequency noise",
                "noise_reduction": {
                    "reduction_strength": 0.6,
                    "freq_mask_smooth_hz": 500,
                    "time_mask_smooth_ms": 70
                }
            },
            "office_noise": {
                "description": "Mixed frequency, keyboard sounds",
                "noise_reduction": {
                    "reduction_strength": 0.6,
                    "freq_mask_smooth_hz": 400,
                    "time_mask_smooth_ms": 30
                }
            },
            "restaurant_chatter": {
                "description": "Speech-like noise, many voices",
                "noise_reduction": {
                    "reduction_strength": 0.5,
                    "freq_mask_smooth_hz": 300,
                    "time_mask_smooth_ms": 50
                }
            },
            "white_noise": {
                "description": "Broadband noise, all frequencies",
                "noise_reduction": {
                    "reduction_strength": 0.8,
                    "freq_mask_smooth_hz": 800,
                    "time_mask_smooth_ms": 120
                }
            }
        }
        
        # Return profile if exists, otherwise return default
        if sound_class in profiles:
            return profiles[sound_class]
        else:
            return {
                "description": "General noise profile",
                "noise_reduction": {
                    "reduction_strength": 0.6,
                    "freq_mask_smooth_hz": 500,
                    "time_mask_smooth_ms": 60
                }
            } 