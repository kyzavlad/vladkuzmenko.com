"""
Voice Activity Detection (VAD) Module

This module provides advanced voice activity detection capabilities
for the Clip Generation Service, allowing precise detection of speech
segments in audio with configurable sensitivity levels.
"""

import os
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union
import tempfile
import torch
from pathlib import Path

from app.clip_generation.services.audio_analysis.audio_analyzer import AudioSegment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VADProcessor:
    """
    Voice Activity Detection processor using state-of-the-art models.
    
    This class implements voice activity detection with configurable
    sensitivity and supports multiple VAD models, including:
    - Silero VAD (default)
    - WebRTC VAD
    - PyAnnote VAD
    
    It provides high-precision speech detection with 50ms granularity.
    """
    
    # VAD model types
    VAD_SILERO = "silero"
    VAD_WEBRTC = "webrtc"
    VAD_PYANNOTE = "pyannote"
    
    def __init__(
        self,
        model_type: str = VAD_SILERO,
        vad_mode: int = 3,  # Aggressiveness level (0-3)
        threshold: float = 0.5,  # Probability threshold for speech
        sampling_rate: int = 16000,  # Required sampling rate
        window_size_ms: int = 96,  # Window size in milliseconds
        min_speech_duration_ms: int = 250,  # Minimum speech duration to keep
        min_silence_duration_ms: int = 300,  # Minimum silence duration to detect
        speech_pad_ms: int = 30,  # Padding around speech segments
        device: str = "cpu",  # Device for inference
        return_seconds: bool = True,  # Return timestamps in seconds
        visualize: bool = False  # Visualize VAD results (for debugging)
    ):
        """
        Initialize the VAD processor.
        
        Args:
            model_type: Type of VAD model to use
            vad_mode: Aggressiveness level (0-3, higher = more aggressive filtering)
            threshold: Probability threshold for speech detection
            sampling_rate: Sampling rate for processing
            window_size_ms: Processing window size in milliseconds
            min_speech_duration_ms: Minimum speech duration to keep
            min_silence_duration_ms: Minimum silence duration to keep
            speech_pad_ms: Padding around speech segments
            device: Device for inference ('cpu' or 'cuda')
            return_seconds: Return timestamps in seconds (True) or samples (False)
            visualize: Visualize VAD results (for debugging)
        """
        self.model_type = model_type
        self.vad_mode = vad_mode
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.window_size_ms = window_size_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.return_seconds = return_seconds
        self.visualize = visualize
        
        # Set device
        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
            logger.info("Using CUDA for VAD processing")
        else:
            self.device = "cpu"
            if device == "cuda":
                logger.warning("CUDA requested but not available, using CPU instead")
            else:
                logger.info("Using CPU for VAD processing")
        
        # Initialize VAD model
        self.model = None
        self.get_speech_timestamps = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the appropriate VAD model based on the selected type."""
        start_time = time.time()
        
        try:
            if self.model_type == self.VAD_SILERO:
                self._load_silero_model()
            elif self.model_type == self.VAD_WEBRTC:
                self._load_webrtc_model()
            elif self.model_type == self.VAD_PYANNOTE:
                self._load_pyannote_model()
            else:
                logger.error(f"Unsupported VAD model type: {self.model_type}")
                raise ValueError(f"Unsupported VAD model type: {self.model_type}")
            
            logger.info(f"Loaded {self.model_type} VAD model in {time.time() - start_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error loading VAD model: {str(e)}")
            raise
    
    def _load_silero_model(self) -> None:
        """Load the Silero VAD model."""
        try:
            # Import torch for Silero VAD
            import torch
            
            logger.info("Loading Silero VAD model")
            
            # Load model from the torch hub
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False
            )
            
            self.model = model.to(self.device)
            self.get_speech_timestamps = utils[0]  # get_speech_timestamps function
            self.collect_chunks = utils[1]         # collect_chunks function
            self.read_audio = utils[2]             # read_audio function
            
            logger.info("Silero VAD model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading Silero VAD model: {str(e)}")
            raise
    
    def _load_webrtc_model(self) -> None:
        """Load the WebRTC VAD model."""
        try:
            # Import webrtcvad for WebRTC VAD
            import webrtcvad
            
            logger.info("Loading WebRTC VAD model")
            
            # Initialize the WebRTC VAD model
            vad = webrtcvad.Vad()
            vad.set_mode(self.vad_mode)
            self.model = vad
            
            logger.info("WebRTC VAD model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading WebRTC VAD model: {str(e)}")
            raise
    
    def _load_pyannote_model(self) -> None:
        """Load the PyAnnote VAD model."""
        try:
            # Import PyAnnote for VAD
            from pyannote.audio import Pipeline
            
            logger.info("Loading PyAnnote VAD model")
            
            # Initialize the PyAnnote VAD model
            pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection",
                                               use_auth_token=os.environ.get("HUGGINGFACE_TOKEN"))
            self.model = pipeline.to(self.device)
            
            logger.info("PyAnnote VAD model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading PyAnnote VAD model: {str(e)}")
            raise
    
    def detect_speech(self, audio_data: np.ndarray, sample_rate: int) -> List[Dict[str, Union[int, float]]]:
        """
        Detect speech segments in the audio data.
        
        Args:
            audio_data: Numpy array with audio samples (mono)
            sample_rate: Sample rate of the audio data
            
        Returns:
            List of dictionaries with speech timestamps (start, end)
        """
        start_time = time.time()
        
        try:
            # Resample audio if needed
            if sample_rate != self.sampling_rate:
                logger.info(f"Resampling audio from {sample_rate}Hz to {self.sampling_rate}Hz")
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=self.sampling_rate)
                sample_rate = self.sampling_rate
            
            # Process based on selected model type
            if self.model_type == self.VAD_SILERO:
                timestamps = self._detect_with_silero(audio_data, sample_rate)
            elif self.model_type == self.VAD_WEBRTC:
                timestamps = self._detect_with_webrtc(audio_data, sample_rate)
            elif self.model_type == self.VAD_PYANNOTE:
                timestamps = self._detect_with_pyannote(audio_data, sample_rate)
            else:
                logger.error(f"Unsupported VAD model type: {self.model_type}")
                raise ValueError(f"Unsupported VAD model type: {self.model_type}")
            
            logger.info(f"Detected {len(timestamps)} speech segments in {time.time() - start_time:.2f}s")
            
            # Visualize results if requested
            if self.visualize:
                self._visualize_vad_results(audio_data, sample_rate, timestamps)
            
            return timestamps
            
        except Exception as e:
            logger.error(f"Error detecting speech: {str(e)}")
            return []
    
    def _detect_with_silero(self, audio_data: np.ndarray, sample_rate: int) -> List[Dict[str, Union[int, float]]]:
        """
        Detect speech using Silero VAD.
        
        Args:
            audio_data: Numpy array with audio samples (mono)
            sample_rate: Sample rate of the audio data
            
        Returns:
            List of dictionaries with speech timestamps (start, end)
        """
        try:
            # Convert to torch tensor
            tensor = torch.from_numpy(audio_data).float().to(self.device)
            
            # Get speech timestamps
            timestamps = self.get_speech_timestamps(
                tensor,
                self.model,
                threshold=self.threshold,
                sampling_rate=sample_rate,
                min_speech_duration_ms=self.min_speech_duration_ms,
                min_silence_duration_ms=self.min_silence_duration_ms,
                speech_pad_ms=self.speech_pad_ms,
                return_seconds=self.return_seconds
            )
            
            # Convert to standard format
            result = []
            for ts in timestamps:
                if self.return_seconds:
                    result.append({
                        "start": ts["start"],
                        "end": ts["end"]
                    })
                else:
                    # Convert samples to seconds
                    result.append({
                        "start": ts["start"] / sample_rate,
                        "end": ts["end"] / sample_rate
                    })
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting speech with Silero: {str(e)}")
            return []
    
    def _detect_with_webrtc(self, audio_data: np.ndarray, sample_rate: int) -> List[Dict[str, Union[int, float]]]:
        """
        Detect speech using WebRTC VAD.
        
        Args:
            audio_data: Numpy array with audio samples (mono)
            sample_rate: Sample rate of the audio data
            
        Returns:
            List of dictionaries with speech timestamps (start, end)
        """
        try:
            # WebRTC VAD requires 16-bit PCM audio
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Frame size in samples (must be 10, 20, or 30 ms for WebRTC VAD)
            frame_duration_ms = 30  # Use 30ms frames
            frame_size = int(sample_rate * frame_duration_ms / 1000)
            
            # Number of frames
            num_frames = len(audio_int16) // frame_size
            
            # Process frames
            is_speech = []
            for i in range(num_frames):
                frame = audio_int16[i * frame_size:(i + 1) * frame_size]
                frame_bytes = frame.tobytes()
                is_speech.append(self.model.is_speech(frame_bytes, sample_rate))
            
            # Find contiguous speech segments
            speech_segments = []
            in_speech = False
            start_frame = 0
            
            for i, speech in enumerate(is_speech):
                if speech and not in_speech:
                    # Start of speech
                    in_speech = True
                    start_frame = i
                elif not speech and in_speech:
                    # End of speech
                    in_speech = False
                    # Duration check
                    duration_ms = (i - start_frame) * frame_duration_ms
                    if duration_ms >= self.min_speech_duration_ms:
                        # Add padding
                        pad_frames = int(self.speech_pad_ms / frame_duration_ms)
                        start_padded = max(0, start_frame - pad_frames)
                        end_padded = min(num_frames, i + pad_frames)
                        
                        speech_segments.append({
                            "start": start_padded * frame_duration_ms / 1000,
                            "end": end_padded * frame_duration_ms / 1000
                        })
            
            # Check for final speech segment
            if in_speech:
                # Duration check
                duration_ms = (num_frames - start_frame) * frame_duration_ms
                if duration_ms >= self.min_speech_duration_ms:
                    # Add padding
                    pad_frames = int(self.speech_pad_ms / frame_duration_ms)
                    start_padded = max(0, start_frame - pad_frames)
                    end_padded = num_frames
                    
                    speech_segments.append({
                        "start": start_padded * frame_duration_ms / 1000,
                        "end": end_padded * frame_duration_ms / 1000
                    })
            
            # Merge overlapping segments
            speech_segments.sort(key=lambda x: x["start"])
            merged_segments = []
            
            for segment in speech_segments:
                if not merged_segments or segment["start"] > merged_segments[-1]["end"]:
                    merged_segments.append(segment)
                else:
                    merged_segments[-1]["end"] = max(merged_segments[-1]["end"], segment["end"])
            
            return merged_segments
            
        except Exception as e:
            logger.error(f"Error detecting speech with WebRTC: {str(e)}")
            return []
    
    def _detect_with_pyannote(self, audio_data: np.ndarray, sample_rate: int) -> List[Dict[str, Union[int, float]]]:
        """
        Detect speech using PyAnnote VAD.
        
        Args:
            audio_data: Numpy array with audio samples (mono)
            sample_rate: Sample rate of the audio data
            
        Returns:
            List of dictionaries with speech timestamps (start, end)
        """
        try:
            # PyAnnote requires a file input, so we'll save the audio temporarily
            import soundfile as sf
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Save audio to temporary file
            sf.write(temp_path, audio_data, sample_rate)
            
            # Process with PyAnnote
            vad_result = self.model(temp_path)
            
            # Extract speech regions
            speech_segments = []
            for speech in vad_result.get_timeline().support():
                if speech.duration >= self.min_speech_duration_ms / 1000:
                    speech_segments.append({
                        "start": speech.start,
                        "end": speech.end
                    })
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            return speech_segments
            
        except Exception as e:
            logger.error(f"Error detecting speech with PyAnnote: {str(e)}")
            return []
    
    def _visualize_vad_results(self, audio_data: np.ndarray, sample_rate: int, 
                              speech_timestamps: List[Dict[str, Union[int, float]]]) -> None:
        """
        Visualize VAD results for debugging.
        
        Args:
            audio_data: Audio data array
            sample_rate: Sample rate of the audio
            speech_timestamps: List of speech segment timestamps
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            
            duration = len(audio_data) / sample_rate
            time = np.linspace(0, duration, len(audio_data))
            
            plt.figure(figsize=(15, 5))
            plt.plot(time, audio_data, color='gray', alpha=0.5)
            
            # Plot speech segments
            for ts in speech_timestamps:
                plt.axvspan(ts["start"], ts["end"], color='green', alpha=0.3)
            
            plt.title(f"VAD Results ({self.model_type})")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.tight_layout()
            
            # Save to temp file
            temp_dir = Path(tempfile.gettempdir()) / "clip_generation" / "vad_visualization"
            temp_dir.mkdir(parents=True, exist_ok=True)
            output_path = temp_dir / f"vad_visualization_{int(time.time())}.png"
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"VAD visualization saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error visualizing VAD results: {str(e)}")
    
    def segments_to_audio_segments(self, speech_timestamps: List[Dict[str, Union[int, float]]]) -> List[AudioSegment]:
        """
        Convert VAD speech timestamps to AudioSegment objects.
        
        Args:
            speech_timestamps: List of dictionaries with speech timestamps
            
        Returns:
            List of AudioSegment objects
        """
        segments = []
        
        # Create speech segments
        for ts in speech_timestamps:
            segments.append(AudioSegment(
                start_time=ts["start"],
                end_time=ts["end"],
                segment_type="speech",
                confidence=0.9,  # Could be actual confidence from VAD model
                metadata={"vad_model": self.model_type}
            ))
        
        return segments
    
    def process_audio(self, audio_data: np.ndarray, sample_rate: int) -> List[AudioSegment]:
        """
        Process audio to detect speech and silence segments.
        
        Args:
            audio_data: Numpy array with audio samples (mono)
            sample_rate: Sample rate of the audio data
            
        Returns:
            List of AudioSegment objects
        """
        # Detect speech segments
        speech_timestamps = self.detect_speech(audio_data, sample_rate)
        speech_segments = self.segments_to_audio_segments(speech_timestamps)
        
        # Create silence segments (gaps between speech)
        silence_segments = []
        duration = len(audio_data) / sample_rate
        
        if speech_segments:
            # Sort by start time
            speech_segments.sort(key=lambda s: s.start_time)
            
            # Add initial silence if needed
            if speech_segments[0].start_time > 0:
                silence_segments.append(AudioSegment(
                    start_time=0,
                    end_time=speech_segments[0].start_time,
                    segment_type="silence",
                    confidence=0.9,
                    metadata={"vad_model": self.model_type}
                ))
            
            # Add silences between speech segments
            for i in range(len(speech_segments) - 1):
                current_end = speech_segments[i].end_time
                next_start = speech_segments[i+1].start_time
                
                if next_start > current_end:
                    silence_segments.append(AudioSegment(
                        start_time=current_end,
                        end_time=next_start,
                        segment_type="silence",
                        confidence=0.9,
                        metadata={"vad_model": self.model_type}
                    ))
            
            # Add final silence if needed
            if speech_segments[-1].end_time < duration:
                silence_segments.append(AudioSegment(
                    start_time=speech_segments[-1].end_time,
                    end_time=duration,
                    segment_type="silence",
                    confidence=0.9,
                    metadata={"vad_model": self.model_type}
                ))
        else:
            # No speech detected, mark entire audio as silence
            silence_segments.append(AudioSegment(
                start_time=0,
                end_time=duration,
                segment_type="silence",
                confidence=0.9,
                metadata={"vad_model": self.model_type}
            ))
        
        # Filter short silences based on min_silence_duration_ms
        min_silence_duration = self.min_silence_duration_ms / 1000
        silence_segments = [s for s in silence_segments if s.duration >= min_silence_duration]
        
        # Combine all segments and sort by start time
        all_segments = speech_segments + silence_segments
        all_segments.sort(key=lambda s: s.start_time)
        
        return all_segments 