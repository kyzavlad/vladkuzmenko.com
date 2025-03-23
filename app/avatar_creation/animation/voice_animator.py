import os
import numpy as np
import torch
import time
import librosa
import json
from typing import Dict, List, Tuple, Optional, Union, Any
import threading
import queue

from app.avatar_creation.face_modeling.utils import get_device, ensure_directory
from app.avatar_creation.animation.avatar_animator import AvatarAnimator

class VoiceAnimator:
    """
    Class for driving facial animations from voice audio.
    Supports real-time lip-sync and emotion detection from audio.
    """
    
    def __init__(self, 
                avatar_animator: AvatarAnimator,
                model_path: Optional[str] = None,
                use_gpu: bool = True,
                smoothing_factor: float = 0.3,
                emotion_detection: bool = False):
        """
        Initialize the voice animator.
        
        Args:
            avatar_animator: AvatarAnimator instance to control
            model_path: Path to pre-trained voice-to-viseme model
            use_gpu: Whether to use GPU acceleration
            smoothing_factor: Factor for temporal smoothing
            emotion_detection: Whether to detect emotions from voice
        """
        self.device = get_device() if use_gpu else torch.device("cpu")
        self.avatar_animator = avatar_animator
        self.model_path = model_path
        self.smoothing_factor = smoothing_factor
        self.emotion_detection = emotion_detection
        
        # Initialize voice-to-viseme model
        self.model = self._initialize_model()
        
        # Standard viseme set (based on MPEG-4 facial animation parameters)
        self.visemes = [
            "sil",        # Silence
            "PP",         # P, B, M
            "FF",         # F, V
            "TH",         # Th
            "DD",         # D, T, N
            "kk",         # K, G
            "CH",         # CH, J, SH
            "SS",         # S, Z
            "nn",         # N, NG
            "RR",         # R
            "aa",         # A
            "E",          # E
            "I",          # I
            "O",          # O
            "U"           # U
        ]
        
        # Mapping from visemes to blendshape weights
        self.viseme_to_blendshape = {
            "sil": {"face_mouth_closed": 1.0},
            "PP": {"face_mouth_closed": 0.9, "face_lips_pressed": 0.8},
            "FF": {"face_mouth_narrow": 0.5, "face_lips_stretched": 0.4, "face_jaw_open": 0.1},
            "TH": {"face_mouth_narrow": 0.3, "face_tongue_up": 0.7, "face_jaw_open": 0.2},
            "DD": {"face_mouth_closed": 0.5, "face_tongue_up": 0.8, "face_jaw_open": 0.2},
            "kk": {"face_mouth_open": 0.4, "face_jaw_open": 0.3, "face_tongue_back": 0.5},
            "CH": {"face_mouth_narrow": 0.7, "face_lips_stretched": 0.5, "face_jaw_open": 0.2},
            "SS": {"face_mouth_narrow": 0.6, "face_lips_stretched": 0.6, "face_jaw_open": 0.1},
            "nn": {"face_mouth_closed": 0.6, "face_tongue_up": 0.5, "face_jaw_open": 0.2},
            "RR": {"face_mouth_open": 0.3, "face_lips_round": 0.3, "face_tongue_up": 0.3, "face_jaw_open": 0.2},
            "aa": {"face_mouth_open": 0.8, "face_jaw_open": 0.7},
            "E": {"face_mouth_open": 0.6, "face_lips_stretched": 0.3, "face_jaw_open": 0.5},
            "I": {"face_mouth_smile": 0.3, "face_lips_stretched": 0.5, "face_jaw_open": 0.2},
            "O": {"face_mouth_open": 0.6, "face_lips_round": 0.8, "face_jaw_open": 0.4},
            "U": {"face_mouth_open": 0.4, "face_lips_round": 0.9, "face_lips_pucker": 0.5, "face_jaw_open": 0.3}
        }
        
        # Current viseme and related state
        self.current_viseme = "sil"
        self.current_viseme_weight = 0.0
        self.viseme_history = []
        self.max_history_length = 30  # frames
        
        # For streaming audio processing
        self.audio_buffer = queue.Queue()
        self.is_processing = False
        self.processing_thread = None
        
        print(f"Voice animator initialized")
        print(f"  - Model: {'Loaded' if self.model else 'None'}")
        print(f"  - Emotion detection: {self.emotion_detection}")
    
    def _initialize_model(self) -> Optional[Any]:
        """
        Initialize the voice-to-viseme model.
        
        Returns:
            Loaded model or None if not available
        """
        if not self.model_path or not os.path.exists(self.model_path):
            print("No voice-to-viseme model specified or found, using rule-based fallback")
            return None
        
        try:
            # This is a placeholder - in a real implementation, you would load
            # a proper deep learning model trained on voice data
            import torch
            
            # Placeholder model loading
            model = torch.nn.Sequential(
                torch.nn.Conv1d(1, 16, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(kernel_size=2),
                torch.nn.Conv1d(16, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(kernel_size=2),
                torch.nn.Flatten(),
                torch.nn.Linear(32 * 250, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, len(self.visemes))
            )
            
            # Try to load pre-trained weights
            try:
                model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                model.to(self.device)
                model.eval()
                print(f"Loaded model from: {self.model_path}")
            except Exception as e:
                print(f"Error loading model weights: {e}")
                print("Using initialized model without pre-trained weights")
            
            return model
            
        except Exception as e:
            print(f"Error initializing model: {e}")
            return None
    
    def process_audio_file(self, audio_path: str, frame_rate: int = 30) -> Dict:
        """
        Process an audio file for lip-sync animation.
        
        Args:
            audio_path: Path to the audio file
            frame_rate: Animation frame rate
            
        Returns:
            Dictionary with animation data
        """
        print(f"Processing audio file: {audio_path}")
        
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Get duration in seconds
            duration = librosa.get_duration(y=audio, sr=sr)
            
            print(f"Audio loaded: {duration:.2f} seconds, {sr} Hz")
            
            # Process audio to extract visemes
            result = self._extract_visemes_from_audio(audio, sr, frame_rate)
            
            # Add metadata
            result['audio_path'] = audio_path
            result['duration'] = duration
            result['sample_rate'] = sr
            result['frame_rate'] = frame_rate
            
            return result
            
        except Exception as e:
            print(f"Error processing audio file: {e}")
            return {'error': str(e)}
    
    def _extract_visemes_from_audio(self, audio: np.ndarray, sample_rate: int, frame_rate: int) -> Dict:
        """
        Extract viseme sequence from audio data.
        
        Args:
            audio: Audio waveform
            sample_rate: Audio sample rate
            frame_rate: Animation frame rate
            
        Returns:
            Dictionary with viseme sequence
        """
        # Calculate frame count
        total_frames = int(len(audio) / sample_rate * frame_rate)
        
        # Initialize viseme frames
        viseme_frames = []
        
        # Time points for each frame
        time_points = np.linspace(0, len(audio) / sample_rate, total_frames)
        
        # If we have a trained model, use it
        if self.model is not None:
            # Prepare batches for the model
            # This is a simplified implementation
            window_size = int(sample_rate * 0.2)  # 200ms window
            hop_length = int(sample_rate / frame_rate)
            
            # Extract features frame by frame
            for i, time_point in enumerate(time_points):
                # Calculate start and end sample
                start_sample = max(0, int(time_point * sample_rate) - window_size // 2)
                end_sample = min(len(audio), start_sample + window_size)
                
                # Extract audio window
                audio_window = audio[start_sample:end_sample]
                
                # If window is too short, pad with zeros
                if len(audio_window) < window_size:
                    audio_window = np.pad(audio_window, (0, window_size - len(audio_window)))
                
                # Process the window
                with torch.no_grad():
                    # Prepare input tensor (example, this would depend on your model)
                    audio_tensor = torch.tensor(audio_window, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
                    
                    # Run the model
                    output = self.model(audio_tensor)
                    
                    # Get predicted viseme
                    viseme_probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                    viseme_idx = np.argmax(viseme_probs)
                    viseme = self.visemes[viseme_idx]
                    confidence = viseme_probs[viseme_idx]
                
                # Add frame data
                viseme_frames.append({
                    'time': time_point,
                    'viseme': viseme,
                    'confidence': float(confidence)
                })
        
        else:
            # Use a simple rule-based approach
            # This is a very simplified implementation
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Extract energy and zero-crossing rate
            hop_length = int(sample_rate / frame_rate)
            energy = librosa.feature.rms(y=audio, frame_length=2048, hop_length=hop_length)[0]
            
            # Normalize energy to 0-1 range
            if len(energy) > 0:
                energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-8)
            
            # Simple mapping based on energy
            for i, time_point in enumerate(time_points):
                frame_idx = min(i, len(energy) - 1)
                
                # Determine viseme based on energy
                if energy[frame_idx] < 0.1:
                    viseme = "sil"  # Silence
                    confidence = 1.0 - energy[frame_idx]
                else:
                    # Choose a non-silent viseme based on energy
                    # This is oversimplified and not linguistically accurate
                    if energy[frame_idx] < 0.3:
                        viseme = "PP"
                        confidence = 0.7
                    elif energy[frame_idx] < 0.5:
                        viseme = "TH"
                        confidence = 0.6
                    elif energy[frame_idx] < 0.7:
                        viseme = "E"
                        confidence = 0.8
                    else:
                        viseme = "aa"
                        confidence = 0.9
                
                # Add frame data
                viseme_frames.append({
                    'time': time_point,
                    'viseme': viseme,
                    'confidence': float(confidence)
                })
                
        # Detect emotions if enabled
        emotions = None
        if self.emotion_detection:
            emotions = self._detect_emotions_from_audio(audio, sample_rate)
        
        return {
            'viseme_frames': viseme_frames,
            'emotions': emotions
        }
    
    def _detect_emotions_from_audio(self, audio: np.ndarray, sample_rate: int) -> List[Dict]:
        """
        Detect emotions from audio data.
        
        Args:
            audio: Audio waveform
            sample_rate: Audio sample rate
            
        Returns:
            List of detected emotions with timestamps
        """
        # This is a simplified implementation
        # In a real implementation, you would use a proper emotion recognition model
        
        # Get audio features
        # Example: Extract pitch, energy, MFCC, etc.
        hop_length = int(sample_rate * 0.1)  # 100ms hop
        
        # Extract energy
        energy = librosa.feature.rms(y=audio, frame_length=2048, hop_length=hop_length)[0]
        
        # Extract pitch (fundamental frequency)
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sample_rate, hop_length=hop_length)
        pitch = []
        for i in range(magnitudes.shape[1]):
            index = magnitudes[:, i].argmax()
            pitch.append(pitches[index, i])
        
        # Calculate deltas
        pitch_delta = np.abs(np.diff(pitch, prepend=pitch[0]))
        energy_delta = np.abs(np.diff(energy, prepend=energy[0]))
        
        # Simple rule-based emotion detection
        emotions = []
        time_step = hop_length / sample_rate
        
        for i in range(len(energy)):
            time_point = i * time_step
            
            # Simplified emotion detection rules
            if energy[i] > 0.8 and pitch_delta[i] > 0.2:
                emotion = "excited"
                confidence = 0.7
            elif energy[i] > 0.6 and pitch[i] > np.mean(pitch) * 1.2:
                emotion = "happy"
                confidence = 0.6
            elif energy[i] < 0.3 and pitch[i] < np.mean(pitch) * 0.8:
                emotion = "sad"
                confidence = 0.5
            elif energy_delta[i] > 0.4:
                emotion = "surprised"
                confidence = 0.6
            elif energy[i] > 0.7 and pitch[i] < np.mean(pitch) * 0.9:
                emotion = "angry"
                confidence = 0.7
            else:
                emotion = "neutral"
                confidence = 0.8
            
            emotions.append({
                'time': time_point,
                'emotion': emotion,
                'confidence': confidence
            })
        
        return emotions
    
    def apply_viseme(self, viseme: str, weight: float = 1.0) -> None:
        """
        Apply a specific viseme to the avatar.
        
        Args:
            viseme: Name of the viseme
            weight: Weight of the viseme (0.0-1.0)
        """
        # Clamp weight to 0-1 range
        weight = max(0.0, min(1.0, weight))
        
        # Update current viseme
        self.current_viseme = viseme
        self.current_viseme_weight = weight
        
        # Add to history
        self.viseme_history.append({
            'timestamp': time.time(),
            'viseme': viseme,
            'weight': weight
        })
        
        # Trim history if needed
        if len(self.viseme_history) > self.max_history_length:
            self.viseme_history.pop(0)
        
        # Apply blendshape weights
        if viseme in self.viseme_to_blendshape:
            # Get blendshape mappings
            blendshapes = self.viseme_to_blendshape[viseme]
            
            # Apply each blendshape
            for blendshape_name, blendshape_weight in blendshapes.items():
                # Apply with the viseme weight
                self.avatar_animator.set_blendshape_weight(
                    blendshape_name, 
                    blendshape_weight * weight
                )
    
    def apply_emotion(self, emotion: str, intensity: float = 1.0) -> None:
        """
        Apply an emotion to the avatar's face.
        
        Args:
            emotion: Name of the emotion
            intensity: Intensity of the emotion (0.0-1.0)
        """
        # Apply emotion-specific expressions
        if emotion == "happy":
            self.avatar_animator.set_blendshape_weight("face_smile", 0.8 * intensity)
            self.avatar_animator.set_blendshape_weight("face_cheek_squint", 0.5 * intensity)
            self.avatar_animator.set_blendshape_weight("face_eye_wide", 0.2 * intensity)
        
        elif emotion == "sad":
            self.avatar_animator.set_blendshape_weight("face_frown", 0.7 * intensity)
            self.avatar_animator.set_blendshape_weight("face_mouth_down", 0.6 * intensity)
            self.avatar_animator.set_blendshape_weight("face_brow_sad", 0.8 * intensity)
        
        elif emotion == "angry":
            self.avatar_animator.set_blendshape_weight("face_brow_lower", 0.8 * intensity)
            self.avatar_animator.set_blendshape_weight("face_nose_wrinkle", 0.6 * intensity)
            self.avatar_animator.set_blendshape_weight("face_jaw_clench", 0.5 * intensity)
        
        elif emotion == "surprised":
            self.avatar_animator.set_blendshape_weight("face_brow_raise", 0.9 * intensity)
            self.avatar_animator.set_blendshape_weight("face_eye_wide", 0.8 * intensity)
            self.avatar_animator.set_blendshape_weight("face_mouth_open", 0.7 * intensity)
        
        elif emotion == "excited":
            self.avatar_animator.set_blendshape_weight("face_smile", 0.9 * intensity)
            self.avatar_animator.set_blendshape_weight("face_eye_wide", 0.7 * intensity)
            self.avatar_animator.set_blendshape_weight("face_brow_raise", 0.5 * intensity)
        
        elif emotion == "neutral":
            # Reset emotion-related blendshapes
            self.avatar_animator.set_blendshape_weight("face_smile", 0.0)
            self.avatar_animator.set_blendshape_weight("face_frown", 0.0)
            self.avatar_animator.set_blendshape_weight("face_brow_raise", 0.0)
            self.avatar_animator.set_blendshape_weight("face_brow_lower", 0.0)
    
    def start_streaming(self, device_index: int = None, sample_rate: int = 16000) -> bool:
        """
        Start streaming audio processing for real-time lip-sync.
        
        Args:
            device_index: Audio input device index (None for default)
            sample_rate: Sample rate for audio processing
            
        Returns:
            True if started successfully, False otherwise
        """
        if self.is_processing:
            print("Streaming already active")
            return False
        
        try:
            # Initialize audio input
            import pyaudio
            p = pyaudio.PyAudio()
            
            # Open stream
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=1024,
                stream_callback=lambda in_data, frame_count, time_info, status_flags: 
                    self._audio_callback(in_data, frame_count, time_info, status_flags)
            )
            
            # Start stream
            stream.start_stream()
            
            # Start processing thread
            self.is_processing = True
            self.processing_thread = threading.Thread(
                target=self._process_audio_stream,
                args=(sample_rate,)
            )
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            print("Audio streaming started")
            return True
            
        except Exception as e:
            print(f"Error starting audio streaming: {e}")
            return False
    
    def _audio_callback(self, in_data, frame_count, time_info, status_flags) -> Tuple[bytes, int]:
        """
        Callback for audio input stream.
        
        Args:
            in_data: Input audio data
            frame_count: Number of frames
            time_info: Time information
            status_flags: Status flags
            
        Returns:
            Tuple of (data, flag)
        """
        try:
            # Convert bytes to float32 array
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # Add to buffer
            self.audio_buffer.put(audio_data)
            
            return (in_data, 0)  # Continue streaming
            
        except Exception as e:
            print(f"Error in audio callback: {e}")
            return (in_data, 0)  # Continue anyway
    
    def _process_audio_stream(self, sample_rate: int) -> None:
        """
        Process audio stream for real-time lip-sync.
        
        Args:
            sample_rate: Audio sample rate
        """
        # Window size for analysis
        window_size = int(sample_rate * 0.1)  # 100ms window
        
        # Buffer for analysis
        analysis_buffer = np.zeros(window_size, dtype=np.float32)
        
        # Processing loop
        while self.is_processing:
            try:
                # Get data from buffer
                try:
                    # Non-blocking get with timeout
                    audio_data = self.audio_buffer.get(timeout=0.1)
                    
                    # Shift buffer and add new data
                    analysis_buffer = np.roll(analysis_buffer, -len(audio_data))
                    analysis_buffer[-len(audio_data):] = audio_data
                    
                except queue.Empty:
                    # No new data
                    pass
                
                # Simple energy-based viseme detection
                energy = np.mean(np.abs(analysis_buffer))
                
                # Determine viseme based on energy
                if energy < 0.01:
                    viseme = "sil"  # Silence
                    weight = 1.0
                else:
                    # Choose a non-silent viseme based on energy
                    # This is oversimplified and not linguistically accurate
                    if energy < 0.05:
                        viseme = "PP"
                        weight = energy * 20
                    elif energy < 0.1:
                        viseme = "TH"
                        weight = energy * 10
                    elif energy < 0.2:
                        viseme = "E"
                        weight = energy * 5
                    else:
                        viseme = "aa"
                        weight = min(energy * 3, 1.0)
                
                # Apply viseme
                self.apply_viseme(viseme, weight)
                
                # Update the avatar
                self.avatar_animator.smoothed_update(self.smoothing_factor)
                
                # Sleep for a short time
                time.sleep(0.03)  # ~30fps
                
            except Exception as e:
                print(f"Error processing audio stream: {e}")
                time.sleep(0.1)  # Sleep to avoid tight loop on error
    
    def stop_streaming(self) -> None:
        """
        Stop streaming audio processing.
        """
        if not self.is_processing:
            print("Streaming not active")
            return
        
        # Stop processing
        self.is_processing = False
        
        # Wait for thread to end
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
            self.processing_thread = None
        
        # Clear buffer
        while not self.audio_buffer.empty():
            try:
                self.audio_buffer.get_nowait()
            except:
                pass
        
        print("Audio streaming stopped")
    
    def create_lipsync_animation(self, audio_path: str, output_dir: str, fps: int = 30) -> Dict:
        """
        Create a lip-sync animation from an audio file.
        
        Args:
            audio_path: Path to the audio file
            output_dir: Directory to save the animation
            fps: Frames per second
            
        Returns:
            Dictionary with animation results
        """
        # Create output directory
        ensure_directory(output_dir)
        
        # Process audio file
        result = self.process_audio_file(audio_path, fps)
        
        if 'error' in result:
            return result
        
        # Create animation data structure
        animation_data = []
        
        # Process each viseme frame
        for i, frame in enumerate(result['viseme_frames']):
            # Create keyframe
            keyframe = {
                'time': frame['time'],
                'blendshapes': {},
                'bones': {}
            }
            
            # Apply viseme
            viseme = frame['viseme']
            weight = frame['confidence']
            
            # Map viseme to blendshapes
            if viseme in self.viseme_to_blendshape:
                for bs_name, bs_weight in self.viseme_to_blendshape[viseme].items():
                    keyframe['blendshapes'][bs_name] = bs_weight * weight
            
            # Apply emotion if available
            if result.get('emotions'):
                # Find closest emotion frame
                closest_idx = min(
                    range(len(result['emotions'])), 
                    key=lambda i: abs(result['emotions'][i]['time'] - frame['time'])
                )
                
                emotion = result['emotions'][closest_idx]
                
                # Add emotion to keyframe
                keyframe['emotion'] = emotion['emotion']
                keyframe['emotion_confidence'] = emotion['confidence']
                
                # Get additional blendshapes for emotion
                # (This would be based on the emotion-to-blendshape mapping)
            
            # Add keyframe to animation data
            animation_data.append(keyframe)
        
        # Create animation sequence
        frame_paths = self.avatar_animator.create_animation_sequence(
            animation_data,
            output_dir,
            fps
        )
        
        # Save animation metadata
        metadata = {
            'audio_path': audio_path,
            'frame_count': len(frame_paths),
            'fps': fps,
            'duration': result['duration'],
            'frames': frame_paths
        }
        
        metadata_path = os.path.join(output_dir, "animation_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            'success': True,
            'frame_count': len(frame_paths),
            'metadata_path': metadata_path,
            'output_dir': output_dir
        } 