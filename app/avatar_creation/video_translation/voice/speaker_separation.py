#!/usr/bin/env python3
"""
Speaker Separation Module for Voice Translation

This module provides functionality for tracking and separating multiple speakers
in audio recordings. It enables voice translation systems to process multi-speaker
content by isolating individual speakers before applying voice translation techniques.

Key features:
- Speaker diarization (who spoke when)
- Multi-speaker voice separation
- Speaker identity preservation
- Overlapping speech handling
- Cross-segment speaker tracking
- Background noise removal
- Gender and age detection
- Speaker embedding extraction
"""

import numpy as np
import librosa
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class SpeakerSegment:
    """A segment of speech from a specific speaker."""
    speaker_id: int  # ID of the speaker
    start_time: float  # Start time in seconds
    end_time: float  # End time in seconds
    text: Optional[str] = None  # Transcript of the segment
    confidence: float = 1.0  # Confidence score for this segment
    audio: Optional[np.ndarray] = None  # Audio data for this segment
    is_overlapping: bool = False  # Whether this segment overlaps with others
    
    @property
    def duration(self) -> float:
        """Get the duration of the segment in seconds."""
        return self.end_time - self.start_time


@dataclass
class SpeakerProfile:
    """Profile information about a speaker."""
    speaker_id: int  # ID of the speaker
    embedding: Optional[np.ndarray] = None  # Speaker embedding vector
    gender: str = "unknown"  # Detected gender: male, female, unknown
    age_range: str = "unknown"  # Detected age range: child, teen, adult, elderly, unknown
    duration: float = 0.0  # Total duration of speech from this speaker
    segments: List[SpeakerSegment] = field(default_factory=list)  # List of segments
    average_pitch: float = 0.0  # Average pitch (F0) of the speaker
    average_energy: float = 0.0  # Average energy of the speaker


class SpeakerSeparator:
    """
    Separates and tracks different speakers in audio recordings.
    """
    
    def __init__(self, model_path: str = "models/speaker/separator_model"):
        """
        Initialize the speaker separator.
        
        Args:
            model_path: Path to the pre-trained speaker separation models
        """
        self.model_path = model_path
        self.loaded = False
        
        # Will be initialized when models are loaded
        self.separation_model = None
        self.diarization_model = None
        self.embedding_model = None
        self.gender_age_detector = None
        
        print(f"Speaker Separator initialized")
        print(f"  - Model path: {model_path}")
    
    def load_models(self) -> bool:
        """
        Load all necessary models for speaker separation.
        
        Returns:
            True if models loaded successfully, False otherwise
        """
        if self.loaded:
            return True
        
        try:
            # In a real implementation, this would load actual models
            # For now, we'll just simulate successful loading
            self.separation_model = object()  # Placeholder for actual model
            self.diarization_model = object()  # Placeholder for actual model
            self.embedding_model = object()  # Placeholder for actual model
            self.gender_age_detector = object()  # Placeholder for actual model
            
            self.loaded = True
            print("Speaker separation models loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading speaker separation models: {e}")
            return False
    
    def separate_speakers(self, 
                        audio_path: str, 
                        num_speakers: Optional[int] = None) -> Dict[int, Tuple[np.ndarray, int]]:
        """
        Separate different speakers from an audio file.
        
        Args:
            audio_path: Path to the audio file
            num_speakers: Number of speakers to separate, or None to auto-detect
            
        Returns:
            Dictionary mapping speaker IDs to tuples of (audio, sample_rate)
        """
        if not self.loaded:
            self.load_models()
        
        # Load audio
        try:
            y, sr = librosa.load(audio_path, sr=None)
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            return {0: (np.array([]), sr)}
        
        # In a real implementation, this would use a sophisticated source separation model
        # For demonstration, we'll create dummy separated audio
        
        # Auto-detect number of speakers if not provided
        if num_speakers is None:
            num_speakers = self._estimate_num_speakers(y, sr)
            print(f"Estimated number of speakers: {num_speakers}")
        
        separated = {}
        for i in range(num_speakers):
            # Create a dummy separation
            # In a real implementation, this would apply proper source separation
            separated[i] = (y * (1.0 / (i + 1)), sr)
        
        return separated
    
    def _estimate_num_speakers(self, audio: np.ndarray, sr: int) -> int:
        """
        Estimate the number of speakers in the audio.
        
        Args:
            audio: Audio signal as numpy array
            sr: Sample rate of the audio
            
        Returns:
            Estimated number of speakers
        """
        # In a real implementation, this would use speaker clustering or diarization
        # For demonstration, we'll return a fixed value
        return 2
    
    def perform_diarization(self, 
                          audio_path: str, 
                          num_speakers: Optional[int] = None) -> List[SpeakerSegment]:
        """
        Determine who spoke when in an audio recording.
        
        Args:
            audio_path: Path to the audio file
            num_speakers: Number of speakers, or None to auto-detect
            
        Returns:
            List of SpeakerSegment objects
        """
        if not self.loaded:
            self.load_models()
        
        # Load audio
        try:
            y, sr = librosa.load(audio_path, sr=None)
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            return []
        
        # In a real implementation, this would use a diarization model
        # For demonstration, we'll create dummy segments
        
        # Auto-detect number of speakers if not provided
        if num_speakers is None:
            num_speakers = self._estimate_num_speakers(y, sr)
        
        # Create segments (simulating alternating speakers)
        segment_duration = 3.0  # 3 seconds per segment
        total_duration = len(y) / sr
        num_segments = int(total_duration / segment_duration)
        
        segments = []
        
        for i in range(num_segments):
            speaker_id = i % num_speakers
            start_time = i * segment_duration
            end_time = min((i + 1) * segment_duration, total_duration)
            
            # Extract segment audio
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment_audio = y[start_sample:end_sample]
            
            segment = SpeakerSegment(
                speaker_id=speaker_id,
                start_time=start_time,
                end_time=end_time,
                confidence=0.9,
                audio=segment_audio
            )
            
            segments.append(segment)
        
        return segments
    
    def extract_speaker_profiles(self, 
                               segments: List[SpeakerSegment],
                               sr: int) -> Dict[int, SpeakerProfile]:
        """
        Extract speaker profiles from diarized segments.
        
        Args:
            segments: List of speaker segments
            sr: Sample rate of the audio
            
        Returns:
            Dictionary mapping speaker IDs to SpeakerProfile objects
        """
        if not self.loaded:
            self.load_models()
        
        # Group segments by speaker
        speaker_segments = defaultdict(list)
        for segment in segments:
            speaker_segments[segment.speaker_id].append(segment)
        
        # Create profiles for each speaker
        profiles = {}
        
        for speaker_id, spk_segments in speaker_segments.items():
            # Calculate total duration
            total_duration = sum(segment.duration for segment in spk_segments)
            
            # Combine all segments for this speaker
            all_audio = []
            for segment in spk_segments:
                if segment.audio is not None:
                    all_audio.append(segment.audio)
            
            if not all_audio:
                continue
                
            combined_audio = np.concatenate(all_audio)
            
            # In a real implementation, these would use actual models
            # For demonstration, we'll create simplified profiles
            
            # Extract speaker embedding
            embedding = np.random.random(128)  # Placeholder for actual embedding
            
            # Detect gender and age
            if combined_audio.size > 0:
                # Simple gender detection based on average pitch
                f0, voiced_flag, _ = librosa.pyin(combined_audio, fmin=60, fmax=500, sr=sr)
                f0_values = f0[voiced_flag] if voiced_flag.any() else np.array([0])
                avg_pitch = np.mean(f0_values) if f0_values.size > 0 else 0
                
                gender = "female" if avg_pitch > 170 else "male"
                
                # Simple age detection (placeholder)
                age_range = "adult"  # Default to adult
                
                # Calculate average energy
                energy = np.mean(librosa.feature.rms(y=combined_audio)[0])
            else:
                avg_pitch = 0
                gender = "unknown"
                age_range = "unknown"
                energy = 0
            
            # Create the profile
            profile = SpeakerProfile(
                speaker_id=speaker_id,
                embedding=embedding,
                gender=gender,
                age_range=age_range,
                duration=total_duration,
                segments=spk_segments,
                average_pitch=float(avg_pitch),
                average_energy=float(energy)
            )
            
            profiles[speaker_id] = profile
        
        return profiles
    
    def enhance_speaker_audio(self, 
                           audio: np.ndarray, 
                           sr: int, 
                           is_speech: bool = True) -> np.ndarray:
        """
        Enhance speaker audio by removing noise and improving quality.
        
        Args:
            audio: Audio signal as numpy array
            sr: Sample rate of the audio
            is_speech: Whether the audio contains speech (True) or not (False)
            
        Returns:
            Enhanced audio signal
        """
        # In a real implementation, this would apply sophisticated enhancement
        # For demonstration, we'll apply basic noise reduction
        
        if audio.size == 0:
            return audio
        
        # Simple noise reduction (high-pass filter to remove low-frequency noise)
        # In a real implementation, this would use more sophisticated methods
        b, a = librosa.filters.butter(4, 100/(sr/2), btype='highpass')
        filtered_audio = librosa.filtfilt(b, a, audio)
        
        return filtered_audio
    
    def handle_overlapping_speech(self, 
                                segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """
        Handle segments with overlapping speech.
        
        Args:
            segments: List of possibly overlapping speaker segments
            
        Returns:
            List of segments with overlapping speech properly handled
        """
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x.start_time)
        
        # Find overlapping segments
        for i, segment in enumerate(sorted_segments):
            # Check if this segment overlaps with any future segments
            for j in range(i + 1, len(sorted_segments)):
                segment2 = sorted_segments[j]
                
                # Check for overlap
                if segment.end_time > segment2.start_time:
                    # Mark both segments as overlapping
                    segment.is_overlapping = True
                    segment2.is_overlapping = True
                    
                # If the second segment starts after this one ends, no need to check further
                if segment2.start_time >= segment.end_time:
                    break
        
        # In a real implementation, this would handle the overlapping segments
        # by applying special separation techniques or creating separate channels
        
        return sorted_segments
    
    def process_audio(self, 
                    audio_path: str,
                    num_speakers: Optional[int] = None) -> Tuple[List[SpeakerSegment], Dict[int, SpeakerProfile]]:
        """
        Process audio to identify and separate speakers.
        
        Args:
            audio_path: Path to the audio file
            num_speakers: Number of speakers, or None to auto-detect
            
        Returns:
            Tuple of (list of speaker segments, dictionary of speaker profiles)
        """
        if not self.loaded:
            self.load_models()
        
        # Step 1: Perform diarization to identify who spoke when
        segments = self.perform_diarization(audio_path, num_speakers)
        
        if not segments:
            return [], {}
        
        # Step 2: Handle overlapping speech
        segments = self.handle_overlapping_speech(segments)
        
        # Get sample rate from the first segment
        sr = librosa.get_samplerate(audio_path)
        
        # Step 3: Extract speaker profiles
        profiles = self.extract_speaker_profiles(segments, sr)
        
        # Step 4: Enhance each speaker's audio
        for segment in segments:
            if segment.audio is not None:
                segment.audio = self.enhance_speaker_audio(segment.audio, sr)
        
        return segments, profiles
    
    def match_speaker_across_recordings(self, 
                                     reference_profile: SpeakerProfile,
                                     comparison_profiles: List[SpeakerProfile]) -> Optional[int]:
        """
        Match a speaker from one recording to speakers in another recording.
        
        Args:
            reference_profile: Profile of the reference speaker
            comparison_profiles: List of profiles to compare against
            
        Returns:
            Speaker ID of the matching speaker, or None if no match found
        """
        if not reference_profile.embedding is not None:
            return None
        
        best_match = None
        best_score = -float('inf')
        
        for profile in comparison_profiles:
            if profile.embedding is not None:
                # Calculate cosine similarity between embeddings
                similarity = np.dot(reference_profile.embedding, profile.embedding) / (
                    np.linalg.norm(reference_profile.embedding) * np.linalg.norm(profile.embedding)
                )
                
                if similarity > 0.7 and similarity > best_score:  # Threshold of 0.7
                    best_score = similarity
                    best_match = profile.speaker_id
        
        return best_match
    
    def mix_separated_audio(self, 
                          separated_audio: Dict[int, Tuple[np.ndarray, int]],
                          segments: List[SpeakerSegment],
                          output_sr: int = 44100) -> np.ndarray:
        """
        Mix separated audio back together according to the segment timing.
        
        Args:
            separated_audio: Dictionary mapping speaker IDs to audio
            segments: List of speaker segments
            output_sr: Desired sample rate for the output
            
        Returns:
            Mixed audio as a numpy array
        """
        # Find the total duration
        total_duration = max(segment.end_time for segment in segments)
        
        # Create an empty output array
        total_samples = int(total_duration * output_sr)
        mixed_audio = np.zeros(total_samples)
        
        # Add each segment at the appropriate time
        for segment in segments:
            if segment.speaker_id in separated_audio:
                speaker_audio, sr = separated_audio[segment.speaker_id]
                
                # If no audio, skip
                if speaker_audio.size == 0:
                    continue
                
                # Resample if necessary
                if sr != output_sr:
                    speaker_audio = librosa.resample(speaker_audio, orig_sr=sr, target_sr=output_sr)
                
                # Calculate start and end samples
                start_sample = int(segment.start_time * output_sr)
                end_sample = int(segment.end_time * output_sr)
                
                # Calculate how many samples we need from the speaker audio
                needed_samples = end_sample - start_sample
                
                # If segment.audio is available, use it directly
                if segment.audio is not None:
                    segment_audio = segment.audio
                    if len(segment_audio) > needed_samples:
                        segment_audio = segment_audio[:needed_samples]
                    elif len(segment_audio) < needed_samples:
                        # Pad with zeros if needed
                        segment_audio = np.pad(segment_audio, (0, needed_samples - len(segment_audio)))
                else:
                    # Otherwise, take a portion from the full speaker audio
                    # In a real implementation, this would be more sophisticated
                    if len(speaker_audio) <= needed_samples:
                        segment_audio = speaker_audio
                        if len(segment_audio) < needed_samples:
                            segment_audio = np.pad(segment_audio, (0, needed_samples - len(segment_audio)))
                    else:
                        # Take a random section
                        start_idx = np.random.randint(0, len(speaker_audio) - needed_samples + 1)
                        segment_audio = speaker_audio[start_idx:start_idx + needed_samples]
                
                # Add to the mixed audio
                end_idx = min(start_sample + len(segment_audio), len(mixed_audio))
                mixed_audio[start_sample:end_idx] += segment_audio[:end_idx - start_sample]
        
        # Normalize to prevent clipping
        if np.max(np.abs(mixed_audio)) > 1.0:
            mixed_audio = mixed_audio / np.max(np.abs(mixed_audio))
        
        return mixed_audio


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    
    # Initialize the speaker separator
    separator = SpeakerSeparator()
    
    # Example audio path (this would be a real file in actual usage)
    example_audio_path = "example/multi_speaker.wav"
    
    try:
        if os.path.exists(example_audio_path):
            # Process the audio
            segments, profiles = separator.process_audio(example_audio_path)
            
            # Print results
            print(f"Found {len(profiles)} speakers:")
            for speaker_id, profile in profiles.items():
                print(f"  Speaker {speaker_id}:")
                print(f"    Gender: {profile.gender}")
                print(f"    Age: {profile.age_range}")
                print(f"    Duration: {profile.duration:.2f}s")
                print(f"    Average pitch: {profile.average_pitch:.2f} Hz")
                print(f"    Segments: {len(profile.segments)}")
            
            # Separate speakers
            separated_audio = separator.separate_speakers(example_audio_path)
            
            # Mix back together
            mixed_audio = separator.mix_separated_audio(separated_audio, segments)
            
            print(f"Successfully separated and mixed {len(separated_audio)} speakers")
            
            # Visualize the segments
            plt.figure(figsize=(12, 6))
            
            # Create a timeline of who spoke when
            for segment in segments:
                plt.barh(
                    segment.speaker_id,
                    segment.duration,
                    left=segment.start_time,
                    height=0.8,
                    color=f"C{segment.speaker_id}",
                    alpha=0.8
                )
            
            plt.xlabel("Time (seconds)")
            plt.ylabel("Speaker ID")
            plt.title("Speaker Diarization Results")
            plt.tight_layout()
            plt.savefig("diarization_results.png")
            
            print("Visualization saved to diarization_results.png")
        else:
            print(f"Example file {example_audio_path} not found. This is just a demonstration.")
            
    except Exception as e:
        print(f"Error processing audio: {e}")
        print("This is just a demonstration. Please use with actual audio files.") 