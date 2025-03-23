"""
Filler Sound Detection Module

This module provides capabilities to detect and identify filler sounds,
hesitations, repetitions and other unnecessary audio segments that can
be removed to improve the quality of the final clip.
"""

import os
import re
import numpy as np
import logging
import time
import json
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from pathlib import Path
import tempfile

from app.clip_generation.services.audio_analysis.audio_analyzer import AudioSegment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FillerWordDetector:
    """
    Detector for filler words and hesitations in speech.
    
    This class identifies common filler words, hesitations,
    repetitions, and other unnecessary sounds in speech that
    can be removed to improve the quality of the content.
    """
    
    # Default filler words by language
    DEFAULT_FILLER_WORDS = {
        "en": [
            "um", "uh", "er", "ah", "like", "you know", "i mean",
            "so", "actually", "basically", "literally", "right",
            "mhm", "hmm", "well", "sort of", "kind of", "okay so",
        ],
        "es": [
            "eh", "este", "em", "pues", "o sea", "bueno", "como",
            "digamos", "entonces", "¿sabes?", "¿vale?", "¿no?",
        ],
        "fr": [
            "euh", "ben", "bah", "tu vois", "en fait", "genre",
            "bon", "bref", "quoi", "voilà", "donc",
        ],
        "de": [
            "äh", "ähm", "öhm", "tja", "halt", "quasi", "sozusagen",
            "also", "eigentlich", "naja", "quasi", "so",
        ],
        "ja": [
            "えーと", "あのー", "そのー", "まあ", "なんか", "ね", "さあ",
        ],
        "zh": [
            "那个", "这个", "然后", "就是", "其实", "那么",
        ],
    }
    
    # Hesitation patterns (regex)
    HESITATION_PATTERNS = [
        r"\b(\w+)(?:\s+\1){1,}\b",  # Word repetition ("the the")
        r"\b(\w{1,2}\s+){3,}\b",     # Short word rapid sequence ("I I I")
        r"\b(\w+)-(\1){1,}\b",       # Stuttering ("s-s-something")
        r"\.{3,}",                   # Long pause "..."
    ]
    
    # Breath and mouth sounds patterns
    BREATH_SOUNDS = [
        "*breath*", "*inhale*", "*exhale*", "*sigh*",
        "*cough*", "*throat*", "*swallow*", "*lip smack*",
        "*mouth noise*"
    ]
    
    def __init__(
        self,
        language: str = "en",
        custom_filler_words: Optional[List[str]] = None,
        custom_patterns: Optional[List[str]] = None,
        filler_word_detection_threshold: float = 0.7,
        repetition_detection_threshold: float = 0.8,
        breath_detection_threshold: float = 0.6,
        enable_custom_detection: bool = False,
        model_dir: Optional[str] = None
    ):
        """
        Initialize the filler word detector.
        
        Args:
            language: Language code for filler word detection
            custom_filler_words: Additional filler words to detect
            custom_patterns: Additional regex patterns for detection
            filler_word_detection_threshold: Confidence threshold for filler words
            repetition_detection_threshold: Confidence threshold for repetition
            breath_detection_threshold: Confidence threshold for breath sounds
            enable_custom_detection: Enable ML-based custom detection
            model_dir: Directory for ML models
        """
        self.language = language.lower()
        self.filler_word_detection_threshold = filler_word_detection_threshold
        self.repetition_detection_threshold = repetition_detection_threshold
        self.breath_detection_threshold = breath_detection_threshold
        self.enable_custom_detection = enable_custom_detection
        
        # Set up filler words for the selected language
        self.filler_words = self._get_filler_words(custom_filler_words)
        
        # Compile regex patterns
        self.patterns = self._compile_patterns(custom_patterns)
        
        # Set up model directory
        self.model_dir = Path(model_dir) if model_dir else None
        
        # ASR model for transcription (lazy loaded)
        self._asr_model = None
        
        logger.info(f"Initialized FillerWordDetector for language: {language}")
        logger.info(f"Configured with {len(self.filler_words)} filler words and {len(self.patterns)} patterns")
    
    def _get_filler_words(self, custom_words: Optional[List[str]] = None) -> List[str]:
        """
        Get the list of filler words for the current language.
        
        Args:
            custom_words: Additional custom filler words
            
        Returns:
            List of filler words
        """
        # Get default words for the language
        default_words = self.DEFAULT_FILLER_WORDS.get(
            self.language, 
            self.DEFAULT_FILLER_WORDS["en"]  # Fallback to English
        )
        
        # Combine with custom words if provided
        if custom_words:
            words = default_words + custom_words
            # Remove duplicates while preserving order
            return list(dict.fromkeys(words))
        
        return default_words
    
    def _compile_patterns(self, custom_patterns: Optional[List[str]] = None) -> List[re.Pattern]:
        """
        Compile regex patterns for filler detection.
        
        Args:
            custom_patterns: Additional custom patterns
            
        Returns:
            List of compiled regex patterns
        """
        patterns = self.HESITATION_PATTERNS.copy()
        
        # Add custom patterns if provided
        if custom_patterns:
            patterns.extend(custom_patterns)
        
        # Compile all patterns
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def _load_asr_model(self) -> bool:
        """
        Load ASR model for speech transcription.
        
        Returns:
            True if successful, False otherwise
        """
        if self._asr_model is not None:
            return True
        
        try:
            import torch
            from transformers import pipeline
            
            logger.info("Loading ASR model for transcription")
            
            # Use WhisperX or Whisper for transcription
            model_name = "openai/whisper-small"
            
            self._asr_model = pipeline(
                "automatic-speech-recognition",
                model=model_name,
                chunk_length_s=30,
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info(f"ASR model {model_name} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading ASR model: {str(e)}")
            return False
    
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Transcribe audio to text using ASR.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Dictionary with transcription results
        """
        if not self._load_asr_model():
            logger.error("ASR model not available for transcription")
            return {"text": "", "chunks": []}
        
        try:
            # Import required libraries
            import soundfile as sf
            
            # Create temporary file for audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                
            # Save audio to temporary file
            sf.write(temp_path, audio_data, sample_rate)
            
            logger.info(f"Transcribing audio ({len(audio_data) / sample_rate:.2f}s)")
            
            # Transcribe with timestamps
            result = self._asr_model(
                temp_path,
                return_timestamps=True
            )
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            # Process result
            if "chunks" in result:
                # Newer Whisper API returns chunks directly
                chunks = result["chunks"]
            else:
                # Older API may return different format
                chunks = []
                if "timestamps" in result:
                    for i, timestamp in enumerate(result["timestamps"]):
                        chunks.append({
                            "text": timestamp[0],
                            "timestamp": (timestamp[1], timestamp[2])
                        })
            
            return {
                "text": result["text"],
                "chunks": chunks
            }
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return {"text": "", "chunks": []}
    
    def detect_filler_words(self, transcript: str) -> List[Dict[str, Any]]:
        """
        Detect filler words in the transcript.
        
        Args:
            transcript: Text transcript to analyze
            
        Returns:
            List of detected filler words with positions
        """
        results = []
        
        # Create regex pattern from filler words
        # Ensure word boundaries for whole words
        pattern_str = r'\b(?:' + '|'.join(map(re.escape, self.filler_words)) + r')\b'
        pattern = re.compile(pattern_str, re.IGNORECASE)
        
        # Find all matches
        for match in pattern.finditer(transcript.lower()):
            results.append({
                "type": "filler_word",
                "text": match.group(),
                "start": match.start(),
                "end": match.end(),
                "confidence": self.filler_word_detection_threshold
            })
        
        return results
    
    def detect_hesitations(self, transcript: str) -> List[Dict[str, Any]]:
        """
        Detect hesitations in the transcript.
        
        Args:
            transcript: Text transcript to analyze
            
        Returns:
            List of detected hesitations with positions
        """
        results = []
        
        # Check for each pattern
        for pattern in self.patterns:
            for match in pattern.finditer(transcript):
                results.append({
                    "type": "hesitation",
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": self.repetition_detection_threshold,
                    "pattern": pattern.pattern
                })
        
        return results
    
    def detect_breath_sounds(self, transcript: str) -> List[Dict[str, Any]]:
        """
        Detect breath and mouth sounds in the transcript.
        
        Args:
            transcript: Text transcript to analyze
            
        Returns:
            List of detected breath sounds with positions
        """
        results = []
        
        # Look for annotated breath sounds
        for breath in self.BREATH_SOUNDS:
            start = 0
            while True:
                start = transcript.find(breath, start)
                if start == -1:
                    break
                    
                end = start + len(breath)
                results.append({
                    "type": "breath_sound",
                    "text": breath,
                    "start": start,
                    "end": end,
                    "confidence": self.breath_detection_threshold
                })
                start = end
        
        return results
    
    def map_detections_to_timestamps(
        self, 
        detections: List[Dict[str, Any]], 
        transcript_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Map text detections to audio timestamps.
        
        Args:
            detections: List of detected items (filler words, hesitations, etc.)
            transcript_chunks: Transcript chunks with timestamps
            
        Returns:
            List of detections with audio timestamps
        """
        if not detections or not transcript_chunks:
            return []
        
        # Build character-to-time mapping
        char_to_time = {}
        char_index = 0
        
        for chunk in transcript_chunks:
            text = chunk.get("text", "")
            start_time, end_time = chunk.get("timestamp", (0, 0))
            
            # Character duration in this chunk
            if len(text) > 0:
                char_duration = (end_time - start_time) / len(text)
                
                # Map each character to its approximate time
                for i, _ in enumerate(text):
                    # Linear interpolation for time
                    time = start_time + i * char_duration
                    char_to_time[char_index + i] = time
                
                # Update character index
                char_index += len(text)
            
            # Add space between chunks if not already present
            if char_index > 0 and char_index - 1 in char_to_time:
                char_to_time[char_index] = end_time
                char_index += 1
        
        # Map detections to timestamps
        results = []
        
        for detection in detections:
            start_idx = detection["start"]
            end_idx = detection["end"]
            
            # Find closest mapped indices
            start_time = None
            end_time = None
            
            # Find start time (search forward from start_idx)
            for i in range(start_idx, min(end_idx, max(char_to_time.keys()) + 1)):
                if i in char_to_time:
                    start_time = char_to_time[i]
                    break
            
            # Find end time (search backward from end_idx)
            for i in range(min(end_idx, max(char_to_time.keys())), start_idx - 1, -1):
                if i in char_to_time:
                    end_time = char_to_time[i]
                    break
            
            # Skip if we couldn't find timestamps
            if start_time is None or end_time is None:
                continue
            
            # Create result with timestamps
            result = detection.copy()
            result["start_time"] = start_time
            result["end_time"] = end_time
            results.append(result)
        
        return results
    
    def detect_fillers_in_audio(self, audio_data: np.ndarray, sample_rate: int) -> List[AudioSegment]:
        """
        Detect filler sounds in audio data.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            List of AudioSegment objects for filler sounds
        """
        # Step 1: Transcribe audio
        transcription = self.transcribe_audio(audio_data, sample_rate)
        transcript = transcription.get("text", "")
        chunks = transcription.get("chunks", [])
        
        if not transcript or not chunks:
            logger.warning("No transcript or chunks available for filler detection")
            return []
        
        # Step 2: Detect various types of unnecessary sounds
        filler_words = self.detect_filler_words(transcript)
        hesitations = self.detect_hesitations(transcript)
        breath_sounds = self.detect_breath_sounds(transcript)
        
        # Combine all detections
        all_detections = filler_words + hesitations + breath_sounds
        
        # Step 3: Map detections to audio timestamps
        timestamped_detections = self.map_detections_to_timestamps(all_detections, chunks)
        
        # Step 4: Convert to AudioSegment objects
        segments = []
        
        for detection in timestamped_detections:
            segment_type = detection["type"]
            
            # Skip if missing time information
            if "start_time" not in detection or "end_time" not in detection:
                continue
                
            # Create audio segment
            segment = AudioSegment(
                start_time=detection["start_time"],
                end_time=detection["end_time"],
                segment_type="filler",  # All are categorized as filler for removal
                confidence=detection.get("confidence", 0.7),
                metadata={
                    "text": detection.get("text", ""),
                    "filler_type": segment_type,
                    "detection_source": "transcription",
                    "removable": True
                }
            )
            
            segments.append(segment)
        
        logger.info(f"Detected {len(segments)} filler sounds in audio")
        
        return segments
    
    def custom_detection(self, audio_data: np.ndarray, sample_rate: int) -> List[AudioSegment]:
        """
        Perform custom ML-based detection for fillers.
        
        This is an extension point for more advanced detection methods.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            List of AudioSegment objects
        """
        if not self.enable_custom_detection:
            return []
        
        try:
            # Placeholder for custom ML-based detection
            # This would require a specialized model for direct audio analysis
            # rather than text-based detection
            logger.info("Custom filler detection is enabled but not yet implemented")
            return []
            
        except Exception as e:
            logger.error(f"Error in custom filler detection: {str(e)}")
            return []
    
    def process_audio(self, audio_data: np.ndarray, sample_rate: int) -> List[AudioSegment]:
        """
        Process audio to detect all filler sounds.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            List of AudioSegment objects for filler sounds
        """
        start_time = time.time()
        logger.info(f"Processing audio for filler sound detection ({len(audio_data) / sample_rate:.2f}s)")
        
        # Perform transcription-based detection
        segments = self.detect_fillers_in_audio(audio_data, sample_rate)
        
        # Add custom detection results if enabled
        if self.enable_custom_detection:
            custom_segments = self.custom_detection(audio_data, sample_rate)
            segments.extend(custom_segments)
        
        # Sort by start time
        segments.sort(key=lambda s: s.start_time)
        
        # Merge overlapping segments
        if len(segments) > 1:
            merged_segments = [segments[0]]
            
            for segment in segments[1:]:
                if segment.overlaps(merged_segments[-1]):
                    merged_segments[-1] = merged_segments[-1].merge(segment)
                else:
                    merged_segments.append(segment)
            
            segments = merged_segments
        
        logger.info(f"Filler detection completed in {time.time() - start_time:.2f}s, found {len(segments)} segments")
        return segments 