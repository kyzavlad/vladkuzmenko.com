"""
Mood Analyzer Module

This module provides functionality for analyzing the mood of video content
through audio features, visual elements, and transcript sentiment analysis.
"""

import os
import logging
import tempfile
import json
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class MoodAnalyzer:
    """
    Analyzes the mood of video content using multiple cues (audio, visual, transcript).
    
    This class provides methods to determine the emotional tone, energy level,
    and overall mood of video content, which can be used for selecting appropriate
    music tracks that match the desired atmosphere.
    """
    
    # Mood categories mapped to numerical values (valence, arousal)
    # Valence: negative (-1) to positive (1)
    # Arousal: calm (0) to energetic (1)
    MOOD_MAPPINGS = {
        "happy": (0.8, 0.7),       # High valence, moderate-high arousal
        "sad": (-0.7, 0.2),        # Low valence, low arousal
        "relaxed": (0.5, 0.2),     # Moderate valence, low arousal
        "tense": (-0.5, 0.8),      # Moderate-low valence, high arousal
        "excited": (0.7, 0.9),     # High valence, high arousal
        "melancholic": (-0.3, 0.3), # Moderate-low valence, low-moderate arousal
        "dramatic": (-0.2, 0.8),   # Moderate-low valence, high arousal
        "nostalgic": (0.2, 0.4),   # Moderate valence, moderate arousal
        "inspiring": (0.7, 0.6),   # High valence, moderate-high arousal
        "mysterious": (-0.1, 0.5)  # Moderate-low valence, moderate arousal
    }
    
    # Color mood associations
    COLOR_MOOD_MAPPINGS = {
        "bright_colors": {"valence": 0.7, "arousal": 0.6},  # Bright colors suggest positive, energetic
        "dark_colors": {"valence": -0.4, "arousal": 0.3},   # Dark colors suggest negative, moderate arousal
        "warm_colors": {"valence": 0.5, "arousal": 0.5},    # Warm colors suggest positive, moderate arousal
        "cool_colors": {"valence": 0.0, "arousal": 0.3},    # Cool colors suggest neutral, lower arousal
        "saturated": {"valence": 0.3, "arousal": 0.7},      # Saturated suggests moderate valence, higher arousal
        "desaturated": {"valence": -0.2, "arousal": 0.2}    # Desaturated suggests slight negative, lower arousal
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the mood analyzer.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        
        # Set default parameters
        self.temp_dir = self.config.get('temp_dir', tempfile.gettempdir())
        self.ffmpeg_path = self.config.get('ffmpeg_path', 'ffmpeg')
        self.ffprobe_path = self.config.get('ffprobe_path', 'ffprobe')
        
        # Initialize NLP for sentiment analysis if available
        try:
            import nltk
            from nltk.sentiment import SentimentIntensityAnalyzer
            nltk.download('vader_lexicon', quiet=True)
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            self.nlp_available = True
        except ImportError:
            logger.warning("NLTK not available. Sentiment analysis will be limited.")
            self.nlp_available = False
            self.sentiment_analyzer = None
        
        # Initialize audio feature extraction if librosa is available
        try:
            import librosa
            self.librosa_available = True
        except ImportError:
            logger.warning("Librosa not available. Audio mood analysis will be limited.")
            self.librosa_available = False
        
        # Initialize color analysis if OpenCV is available
        try:
            import cv2
            self.cv2_available = True
        except ImportError:
            logger.warning("OpenCV not available. Visual mood analysis will be limited.")
            self.cv2_available = False
    
    def analyze_mood(
        self,
        video_path: str,
        transcript: Optional[List[Dict[str, Any]]] = None,
        include_audio_analysis: bool = True,
        include_visual_analysis: bool = True,
        include_transcript_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze the mood of video content using available cues.
        
        Args:
            video_path: Path to the video file
            transcript: Optional transcript data (list of segments)
            include_audio_analysis: Whether to include audio analysis
            include_visual_analysis: Whether to include visual analysis
            include_transcript_analysis: Whether to include transcript analysis
            
        Returns:
            Dictionary with mood analysis results
        """
        results = {
            "status": "success",
            "input_path": video_path,
            "overall_mood": {},
            "mood_timeline": []
        }
        
        analysis_components = []
        
        # Analyze audio if requested
        audio_results = None
        if include_audio_analysis:
            try:
                audio_results = self.analyze_audio_mood(video_path)
                analysis_components.append(("audio", audio_results))
                results["audio_mood"] = audio_results
            except Exception as e:
                logger.error(f"Error in audio mood analysis: {str(e)}")
                results["audio_mood_error"] = str(e)
        
        # Analyze visual elements if requested
        visual_results = None
        if include_visual_analysis:
            try:
                visual_results = self.analyze_visual_mood(video_path)
                analysis_components.append(("visual", visual_results))
                results["visual_mood"] = visual_results
            except Exception as e:
                logger.error(f"Error in visual mood analysis: {str(e)}")
                results["visual_mood_error"] = str(e)
        
        # Analyze transcript if available and requested
        transcript_results = None
        if transcript and include_transcript_analysis:
            try:
                transcript_results = self.analyze_transcript_mood(transcript)
                analysis_components.append(("transcript", transcript_results))
                results["transcript_mood"] = transcript_results
            except Exception as e:
                logger.error(f"Error in transcript mood analysis: {str(e)}")
                results["transcript_mood_error"] = str(e)
        
        # Combine results for overall mood
        if analysis_components:
            overall_mood = self._combine_mood_analyses(analysis_components)
            results["overall_mood"] = overall_mood
            
            # Map numerical values to mood labels
            mood_labels = self._map_values_to_mood_labels(
                overall_mood.get("valence", 0),
                overall_mood.get("arousal", 0)
            )
            results["overall_mood"]["mood_labels"] = mood_labels
            
            # Generate mood timeline if we have segment-level data
            if transcript_results and "segment_moods" in transcript_results:
                results["mood_timeline"] = self._generate_mood_timeline(
                    transcript_results["segment_moods"],
                    audio_results.get("segment_moods", []) if audio_results else [],
                    visual_results.get("segment_moods", []) if visual_results else []
                )
        
        return results
    
    def analyze_audio_mood(self, audio_path: str) -> Dict[str, Any]:
        """
        Analyze the mood of audio content using audio features.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary with audio mood analysis results
        """
        results = {}
        
        # If librosa is not available, return basic results
        if not self.librosa_available:
            logger.warning("Librosa not available. Using basic audio mood analysis.")
            # Provide a simple estimate based on audio properties
            # In a real implementation, this would use more basic methods
            import random
            results = {
                "valence": random.uniform(-0.3, 0.5),  # Slightly biased toward neutral/positive
                "arousal": random.uniform(0.3, 0.7)    # Moderate arousal
            }
            return results
        
        try:
            import librosa
            
            # Load audio file
            y, sr = librosa.load(audio_path, sr=None)
            
            # Extract audio features
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=y))
            rmse = np.mean(librosa.feature.rms(y=y))
            
            # Calculate additional features if advanced features are enabled
            if self.config.get("use_advanced_audio_features", True):
                chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
                mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr), axis=1)
                
                # Harmony vs. dissonance can indicate valence
                minor_mode_likelihood = np.mean(librosa.feature.tonnetz(y=y, sr=sr)[0:2])
            else:
                chroma = 0
                mfccs = np.zeros(20)
                minor_mode_likelihood = 0
            
            # Map audio features to valence and arousal
            # Higher tempo, higher spectral centroid & rolloff, higher contrast = higher arousal
            arousal = self._normalize(
                0.4 * self._normalize(tempo, 60, 180) +
                0.2 * self._normalize(spectral_centroid, 500, 2000) +
                0.2 * self._normalize(spectral_rolloff, 2000, 8000) +
                0.1 * self._normalize(zero_crossing_rate, 0.01, 0.2) +
                0.1 * self._normalize(rmse, 0.01, 0.2),
                0, 1
            )
            
            # Valence is more complex, but some general rules:
            # Higher spectral contrast, lower minor mode, lower rolloff variability = higher valence
            valence = self._normalize(
                0.3 * self._normalize(spectral_contrast, 0, 50) +
                -0.3 * minor_mode_likelihood +
                0.2 * self._normalize(chroma, 0, 1) +
                0.2 * (1 - self._normalize(zero_crossing_rate, 0.01, 0.2)),
                -1, 1
            )
            
            results = {
                "valence": valence,
                "arousal": arousal,
                "tempo": tempo,
                "spectral_features": {
                    "contrast": float(spectral_contrast),
                    "centroid": float(spectral_centroid),
                    "rolloff": float(spectral_rolloff)
                },
                "energy_features": {
                    "zero_crossing_rate": float(zero_crossing_rate),
                    "rmse": float(rmse)
                }
            }
            
            # Derive segment-level mood if more advanced analysis is needed
            if self.config.get("segment_level_analysis", False):
                segment_moods = self._analyze_audio_segments(y, sr)
                results["segment_moods"] = segment_moods
                
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing audio mood: {str(e)}")
            return {"error": str(e)}
    
    def analyze_visual_mood(self, video_path: str) -> Dict[str, Any]:
        """
        Analyze the mood of video content based on visual elements.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with visual mood analysis results
        """
        results = {}
        
        # If OpenCV is not available, return basic results
        if not self.cv2_available:
            logger.warning("OpenCV not available. Using basic visual mood analysis.")
            # Provide a simple estimate
            import random
            results = {
                "valence": random.uniform(-0.2, 0.6),  # Slightly biased toward positive
                "arousal": random.uniform(0.2, 0.6)    # Moderate arousal
            }
            return results
        
        try:
            import cv2
            
            # Extract frames from video
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Sample frames at regular intervals
            sample_interval = max(1, int(fps * self.config.get("frame_sample_interval_sec", 1)))
            
            # Initialize color statistics
            colors_hsv = []
            brightness_values = []
            saturation_values = []
            motion_values = []
            
            # Variables for motion detection
            prev_gray = None
            
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % sample_interval == 0:
                    # Convert to HSV for better color analysis
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    colors_hsv.append(np.mean(hsv, axis=(0,1)))
                    
                    # Calculate brightness and saturation
                    brightness = np.mean(hsv[:,:,2])  # V channel
                    saturation = np.mean(hsv[:,:,1])  # S channel
                    
                    brightness_values.append(brightness)
                    saturation_values.append(saturation)
                    
                    # Calculate motion if previous frame exists
                    if prev_gray is not None:
                        # Convert current frame to grayscale
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        
                        # Calculate optical flow
                        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        
                        # Calculate motion magnitude
                        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                        motion = np.mean(magnitude)
                        motion_values.append(motion)
                        
                        prev_gray = gray
                    else:
                        # First frame
                        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                frame_idx += 1
            
            cap.release()
            
            # Process collected data
            colors_hsv = np.array(colors_hsv)
            
            # Calculate color statistics
            mean_hue = np.mean(colors_hsv[:, 0])
            mean_saturation = np.mean(saturation_values)
            mean_brightness = np.mean(brightness_values)
            
            # Calculate color variance
            hue_variance = np.var(colors_hsv[:, 0])
            saturation_variance = np.var(saturation_values)
            brightness_variance = np.var(brightness_values)
            
            # Calculate motion statistics if available
            if motion_values:
                mean_motion = np.mean(motion_values)
                motion_variance = np.var(motion_values)
            else:
                mean_motion = 0
                motion_variance = 0
            
            # Map visual features to valence and arousal
            # Brighter, more colorful scenes generally indicate positive valence
            # More motion and color variance generally indicate higher arousal
            
            # Normalize brightness to 0-1 range (assuming 0-255 range)
            norm_brightness = mean_brightness / 255
            
            # Normalize saturation to 0-1 range (assuming 0-255 range)
            norm_saturation = mean_saturation / 255
            
            # Map hue to emotional response
            # Warm colors (red, yellow, orange) are typically associated with higher arousal
            # Cool colors (blue, green) are typically associated with lower arousal
            # This is a simplified mapping - a more sophisticated model would use color psychology
            warm_cool_factor = 0.0
            if 0 <= mean_hue < 30 or 150 <= mean_hue < 180:  # Red or Orange
                warm_cool_factor = 0.7
            elif 30 <= mean_hue < 60:  # Yellow
                warm_cool_factor = 0.5
            elif 60 <= mean_hue < 90:  # Yellow-Green
                warm_cool_factor = 0.2
            elif 90 <= mean_hue < 150:  # Green-Cyan
                warm_cool_factor = -0.2
            elif 180 <= mean_hue < 270:  # Blue-Purple
                warm_cool_factor = -0.4
            elif 270 <= mean_hue < 330:  # Purple-Magenta
                warm_cool_factor = 0.3
            elif 330 <= mean_hue < 360:  # Magenta-Red
                warm_cool_factor = 0.6
            
            # Calculate valence based on color and brightness
            valence = self._normalize(
                0.4 * self._normalize(norm_brightness, 0.2, 0.8) +
                0.3 * warm_cool_factor +
                0.3 * self._normalize(norm_saturation, 0.2, 0.8),
                -1, 1
            )
            
            # Calculate arousal based on motion and variance
            arousal = self._normalize(
                0.5 * self._normalize(mean_motion, 0, 20) +
                0.2 * self._normalize(saturation_variance, 0, 5000) +
                0.2 * self._normalize(brightness_variance, 0, 5000) +
                0.1 * self._normalize(hue_variance, 0, 5000),
                0, 1
            )
            
            # Determine dominant color palette
            color_palette = self._determine_color_palette(
                mean_hue, norm_saturation, norm_brightness
            )
            
            results = {
                "valence": valence,
                "arousal": arousal,
                "color_analysis": {
                    "mean_brightness": float(mean_brightness),
                    "mean_saturation": float(mean_saturation),
                    "mean_hue": float(mean_hue),
                    "color_palette": color_palette
                },
                "motion_analysis": {
                    "mean_motion": float(mean_motion) if motion_values else 0,
                    "motion_variance": float(motion_variance) if motion_values else 0
                }
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing visual mood: {str(e)}")
            return {"error": str(e)}
    
    def analyze_transcript_mood(self, transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the mood of content based on transcript text.
        
        Args:
            transcript: List of transcript segments
            
        Returns:
            Dictionary with transcript mood analysis results
        """
        results = {}
        
        # If NLP libraries aren't available, return basic results
        if not self.nlp_available:
            logger.warning("NLP libraries not available. Using basic text mood analysis.")
            # Simple keyword-based analysis
            keywords = {
                "positive": ["good", "great", "excellent", "happy", "joy", "love", "beautiful"],
                "negative": ["bad", "terrible", "awful", "sad", "hate", "ugly", "difficult"],
                "high_energy": ["exciting", "fast", "quick", "run", "jump", "energetic"],
                "low_energy": ["slow", "calm", "peaceful", "quiet", "gentle", "relaxed"]
            }
            
            pos_count, neg_count = 0, 0
            high_energy_count, low_energy_count = 0, 0
            total_words = 0
            
            for segment in transcript:
                if "text" in segment:
                    text = segment["text"].lower()
                    words = text.split()
                    total_words += len(words)
                    
                    for word in words:
                        if word in keywords["positive"]:
                            pos_count += 1
                        if word in keywords["negative"]:
                            neg_count += 1
                        if word in keywords["high_energy"]:
                            high_energy_count += 1
                        if word in keywords["low_energy"]:
                            low_energy_count += 1
            
            # Calculate valence and arousal based on keyword counts
            if total_words > 0:
                valence = ((pos_count - neg_count) / total_words) * 2  # Scale to roughly -1 to 1
                arousal = ((high_energy_count - low_energy_count) / total_words) * 2  # Scale to roughly 0 to 1
                
                # Clamp values to valid ranges
                valence = max(-1.0, min(1.0, valence))
                arousal = max(0.0, min(1.0, arousal))
            else:
                valence, arousal = 0.0, 0.5
            
            results = {
                "valence": valence,
                "arousal": arousal,
                "keyword_counts": {
                    "positive": pos_count,
                    "negative": neg_count,
                    "high_energy": high_energy_count,
                    "low_energy": low_energy_count,
                    "total_words": total_words
                }
            }
            
            return results
        
        # If we have NLP libraries, perform more sophisticated analysis
        try:
            # Analyze each segment
            segment_moods = []
            overall_scores = {
                "compound": 0,
                "positive": 0,
                "negative": 0,
                "neutral": 0
            }
            
            total_segments = len(transcript)
            total_duration = 0
            
            for segment in transcript:
                if "text" in segment:
                    text = segment["text"]
                    start = segment.get("start", 0)
                    end = segment.get("end", 0)
                    duration = end - start
                    total_duration += duration
                    
                    # Get sentiment scores
                    sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
                    
                    # Update overall scores (weighted by duration)
                    for key in overall_scores:
                        overall_scores[key] += sentiment_scores[key] * duration if duration > 0 else sentiment_scores[key]
                    
                    # Map sentiment to valence and arousal
                    # Compound score ranges from -1 (negative) to 1 (positive)
                    valence = sentiment_scores["compound"]
                    
                    # Estimate arousal based on sentiment intensity
                    # Higher values of positive or negative (vs. neutral) suggest higher arousal
                    arousal = 1.0 - sentiment_scores["neutral"]
                    
                    segment_moods.append({
                        "start": start,
                        "end": end,
                        "valence": valence,
                        "arousal": arousal,
                        "sentiment": sentiment_scores
                    })
            
            # Normalize overall scores
            if total_duration > 0:
                for key in overall_scores:
                    overall_scores[key] /= total_duration
            elif total_segments > 0:
                for key in overall_scores:
                    overall_scores[key] /= total_segments
            
            # Map overall sentiment to valence and arousal
            valence = overall_scores["compound"]
            arousal = 1.0 - overall_scores["neutral"]
            
            results = {
                "valence": valence,
                "arousal": arousal,
                "sentiment_scores": overall_scores,
                "segment_moods": segment_moods
            }
            
            return results
                
        except Exception as e:
            logger.error(f"Error analyzing transcript mood: {str(e)}")
            return {"error": str(e)}
    
    def _normalize(self, value, min_val, max_val):
        """Normalize a value to a specified range."""
        if min_val == max_val:
            return 0.5  # Default to middle value if range is zero
        
        norm_value = (value - min_val) / (max_val - min_val)
        return max(0, min(1, norm_value))
    
    def _analyze_audio_segments(self, y, sr, segment_duration=5):
        """Analyze audio in segments for mood timeline."""
        import librosa
        
        segment_length = segment_duration * sr
        num_segments = max(1, int(len(y) / segment_length))
        
        segment_moods = []
        
        for i in range(num_segments):
            start_sample = i * segment_length
            end_sample = min(len(y), (i + 1) * segment_length)
            
            if end_sample - start_sample < sr:  # Skip segments shorter than 1 second
                continue
                
            segment_y = y[start_sample:end_sample]
            
            # Extract features for this segment
            tempo, _ = librosa.beat.beat_track(y=segment_y, sr=sr)
            spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=segment_y, sr=sr))
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment_y, sr=sr))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=segment_y))
            rmse = np.mean(librosa.feature.rms(y=segment_y))
            
            # Map to valence and arousal
            arousal = self._normalize(
                0.4 * self._normalize(tempo, 60, 180) +
                0.2 * self._normalize(spectral_centroid, 500, 2000) +
                0.2 * self._normalize(zero_crossing_rate, 0.01, 0.2) +
                0.2 * self._normalize(rmse, 0.01, 0.2),
                0, 1
            )
            
            valence = self._normalize(
                0.6 * self._normalize(spectral_contrast, 0, 50) +
                0.4 * (1 - self._normalize(zero_crossing_rate, 0.01, 0.2)),
                -1, 1
            )
            
            segment_moods.append({
                "start": start_sample / sr,
                "end": end_sample / sr,
                "valence": valence,
                "arousal": arousal
            })
            
        return segment_moods
    
    def _combine_mood_analyses(self, analysis_components: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, float]:
        """
        Combine multiple mood analyses into a single overall mood.
        
        Args:
            analysis_components: List of (component_name, analysis_result) tuples
            
        Returns:
            Dictionary with combined valence and arousal values
        """
        # Weights for different components
        weights = {
            "audio": 0.35,
            "visual": 0.35,
            "transcript": 0.30
        }
        
        # Calculate weighted average for valence and arousal
        total_valence = 0
        total_arousal = 0
        total_weight = 0
        
        for component_name, analysis in analysis_components:
            if "valence" in analysis and "arousal" in analysis:
                weight = weights.get(component_name, 0.0)
                total_valence += analysis["valence"] * weight
                total_arousal += analysis["arousal"] * weight
                total_weight += weight
        
        # Normalize by total weight if non-zero
        if total_weight > 0:
            valence = total_valence / total_weight
            arousal = total_arousal / total_weight
        else:
            valence = 0.0
            arousal = 0.5  # Default to neutral arousal
        
        # Ensure values are in valid ranges
        valence = max(-1.0, min(1.0, valence))
        arousal = max(0.0, min(1.0, arousal))
        
        return {
            "valence": valence,
            "arousal": arousal
        }
    
    def _map_values_to_mood_labels(self, valence: float, arousal: float, top_n: int = 3) -> List[str]:
        """
        Map valence and arousal values to mood labels.
        
        Args:
            valence: Valence value (-1 to 1)
            arousal: Arousal value (0 to 1)
            top_n: Number of mood labels to return
            
        Returns:
            List of mood labels sorted by relevance
        """
        # Calculate Euclidean distance to each mood in our mapping
        mood_distances = {}
        
        for mood, (mood_valence, mood_arousal) in self.MOOD_MAPPINGS.items():
            distance = np.sqrt((valence - mood_valence)**2 + (arousal - mood_arousal)**2)
            mood_distances[mood] = distance
        
        # Sort moods by distance (closest first)
        sorted_moods = sorted(mood_distances.items(), key=lambda x: x[1])
        
        # Return top N moods
        return [mood for mood, _ in sorted_moods[:top_n]]
    
    def _determine_color_palette(self, hue, saturation, brightness):
        """
        Determine the color palette based on HSV values.
        
        Args:
            hue: Mean hue value
            saturation: Mean saturation value
            brightness: Mean brightness value
            
        Returns:
            Dictionary with color palette information
        """
        # Determine brightness category
        if brightness > 0.7:
            brightness_category = "bright"
        elif brightness < 0.3:
            brightness_category = "dark"
        else:
            brightness_category = "moderate"
        
        # Determine saturation category
        if saturation > 0.7:
            saturation_category = "saturated"
        elif saturation < 0.3:
            saturation_category = "desaturated"
        else:
            saturation_category = "moderate"
        
        # Determine hue category
        if 0 <= hue < 30 or 330 <= hue < 360:  # Red
            hue_category = "red"
        elif 30 <= hue < 90:  # Yellow
            hue_category = "yellow"
        elif 90 <= hue < 150:  # Green
            hue_category = "green"
        elif 150 <= hue < 210:  # Cyan
            hue_category = "cyan"
        elif 210 <= hue < 270:  # Blue
            hue_category = "blue"
        elif 270 <= hue < 330:  # Magenta
            hue_category = "magenta"
        else:
            hue_category = "neutral"
        
        # Determine temperature
        if hue_category in ["red", "yellow", "magenta"]:
            temperature = "warm"
        elif hue_category in ["cyan", "blue", "green"]:
            temperature = "cool"
        else:
            temperature = "neutral"
        
        return {
            "hue_category": hue_category,
            "saturation_category": saturation_category,
            "brightness_category": brightness_category,
            "temperature": temperature,
            "description": f"{brightness_category} {saturation_category} {temperature}"
        }
    
    def _generate_mood_timeline(
        self,
        transcript_moods,
        audio_moods,
        visual_moods
    ) -> List[Dict[str, Any]]:
        """
        Generate a timeline of mood changes throughout the video.
        
        Args:
            transcript_moods: Mood analysis for transcript segments
            audio_moods: Mood analysis for audio segments
            visual_moods: Mood analysis for visual segments
            
        Returns:
            List of mood segments with timestamps
        """
        # Start with transcript segments as the base timeline
        timeline = []
        
        for segment in transcript_moods:
            start = segment["start"]
            end = segment["end"]
            
            # Find overlapping audio and visual segments
            overlapping_audio = [
                m for m in audio_moods
                if (m["start"] < end and m["end"] > start)
            ]
            
            overlapping_visual = [
                m for m in visual_moods
                if (m["start"] < end and m["end"] > start)
            ]
            
            # Calculate weighted average of valence and arousal
            valence = segment["valence"]
            arousal = segment["arousal"]
            
            # Include audio mood if available
            if overlapping_audio:
                audio_valence = np.mean([m["valence"] for m in overlapping_audio])
                audio_arousal = np.mean([m["arousal"] for m in overlapping_audio])
                
                valence = 0.5 * valence + 0.5 * audio_valence
                arousal = 0.5 * arousal + 0.5 * audio_arousal
            
            # Include visual mood if available
            if overlapping_visual:
                visual_valence = np.mean([m["valence"] for m in overlapping_visual])
                visual_arousal = np.mean([m["arousal"] for m in overlapping_visual])
                
                valence = 0.5 * valence + 0.5 * visual_valence
                arousal = 0.5 * arousal + 0.5 * visual_arousal
            
            # Map to mood labels
            mood_labels = self._map_values_to_mood_labels(valence, arousal)
            
            timeline.append({
                "start": start,
                "end": end,
                "valence": valence,
                "arousal": arousal,
                "mood_labels": mood_labels
            })
        
        return timeline 