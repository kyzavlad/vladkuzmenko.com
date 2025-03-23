from typing import Tuple, Optional, List
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, vfx
from pydub import AudioSegment
import cv2
import os
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter

class ClipGeneratorError(Exception):
    """Base exception for ClipGenerator errors."""
    pass

class InvalidVideoError(ClipGeneratorError):
    """Raised when the input video is invalid or corrupted."""
    pass

class ProcessingError(ClipGeneratorError):
    """Raised when there's an error during video processing."""
    pass

class AudioProcessingError(ClipGeneratorError):
    """Raised when there's an error during audio processing."""
    pass

class ClipGenerator:
    def __init__(self, input_path: str):
        """
        Initialize the ClipGenerator with an input video path.
        
        Args:
            input_path (str): Path to the input video file
            
        Raises:
            InvalidVideoError: If the video file is invalid or cannot be loaded
        """
        if not os.path.exists(input_path):
            raise InvalidVideoError(f"Video file not found: {input_path}")
            
        try:
            self.input_path = input_path
            self.video = VideoFileClip(input_path)
            self.audio = self.video.audio
            
            # Validate video properties
            if self.video.duration < 1.0:
                raise InvalidVideoError("Video duration is too short")
            if self.video.w < 100 or self.video.h < 100:
                raise InvalidVideoError("Video resolution is too low")
                
        except Exception as e:
            raise InvalidVideoError(f"Failed to load video: {str(e)}")
    
    def _validate_duration(self, duration: float) -> float:
        """
        Validate and adjust target duration.
        
        Args:
            duration (float): Target duration in seconds
            
        Returns:
            float: Validated duration
            
        Raises:
            ValueError: If duration is invalid
        """
        if duration < 5.0:
            raise ValueError("Target duration must be at least 5 seconds")
        if duration > 120.0:
            raise ValueError("Target duration cannot exceed 120 seconds")
        return max(5.0, min(60.0, duration))
    
    def _validate_dimensions(self, width: int, height: int) -> Tuple[int, int]:
        """
        Validate and adjust target dimensions.
        
        Args:
            width (int): Target width in pixels
            height (int): Target height in pixels
            
        Returns:
            Tuple[int, int]: Validated dimensions
            
        Raises:
            ValueError: If dimensions are invalid
        """
        if width < 100 or height < 100:
            raise ValueError("Dimensions must be at least 100x100 pixels")
        if width > 3840 or height > 2160:
            raise ValueError("Dimensions cannot exceed 4K resolution")
        return width, height
    
    def _validate_lufs(self, lufs: float) -> float:
        """
        Validate target LUFS level.
        
        Args:
            lufs (float): Target LUFS level
            
        Returns:
            float: Validated LUFS level
            
        Raises:
            ValueError: If LUFS level is invalid
        """
        if lufs < -31.0 or lufs > -6.0:
            raise ValueError("LUFS level must be between -31 and -6")
        return lufs
    
    def _analyze_scene_changes(self, threshold: float = 30.0) -> List[float]:
        """
        Analyze video for significant scene changes.
        
        Args:
            threshold (float): Threshold for scene change detection
            
        Returns:
            List[float]: List of timestamps where scene changes occur
        """
        scene_changes = []
        prev_frame = None
        
        for t in np.arange(0, self.video.duration, 0.5):
            frame = self.video.get_frame(t)
            if prev_frame is not None:
                # Calculate frame difference
                diff = np.mean(np.abs(frame - prev_frame))
                if diff > threshold:
                    scene_changes.append(t)
            prev_frame = frame
            
        return scene_changes
    
    def _analyze_audio_energy(self) -> List[float]:
        """
        Analyze audio energy to find interesting moments.
        
        Returns:
            List[float]: List of timestamps with high audio energy
        """
        if self.audio is None:
            return []
            
        # Extract audio array
        audio_array = self.audio.to_soundarray()
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)
            
        # Calculate energy
        energy = np.abs(audio_array)
        energy = gaussian_filter(energy, sigma=100)
        
        # Find peaks
        peaks, _ = find_peaks(energy, distance=1000)
        timestamps = peaks / self.audio.fps
        
        return timestamps.tolist()
    
    def _create_transition(self, clip1: VideoFileClip, clip2: VideoFileClip, 
                          duration: float = 0.5) -> VideoFileClip:
        """
        Create a smooth transition between two clips.
        
        Args:
            clip1 (VideoFileClip): First clip
            clip2 (VideoFileClip): Second clip
            duration (float): Transition duration in seconds
            
        Returns:
            VideoFileClip: Combined clips with transition
        """
        # Fade out first clip
        clip1 = clip1.fadeout(duration)
        
        # Fade in second clip
        clip2 = clip2.fadein(duration)
        
        # Concatenate clips
        return concatenate_videoclips([clip1, clip2])
    
    def optimize_duration(self, target_duration: float = 30.0) -> VideoFileClip:
        """
        Optimize the clip duration to target length (default 30s).
        
        Args:
            target_duration (float): Target duration in seconds (5-60s)
            
        Returns:
            VideoFileClip: Optimized video clip
        """
        # Ensure target duration is within bounds
        target_duration = self._validate_duration(target_duration)
        
        # If video is shorter than target, return as is
        if self.video.duration <= target_duration:
            return self.video
            
        # Analyze video for interesting segments
        scene_changes = self._analyze_scene_changes()
        audio_peaks = self._analyze_audio_energy()
        
        # Combine and sort all potential cut points
        cut_points = sorted(set(scene_changes + audio_peaks))
        
        if not cut_points:
            # If no cut points found, use simple segmentation
            return self.video.subclip(0, target_duration)
            
        # Select segments that fit within target duration
        segments = []
        current_duration = 0
        
        for i in range(len(cut_points) - 1):
            start = cut_points[i]
            end = cut_points[i + 1]
            segment_duration = end - start
            
            if current_duration + segment_duration <= target_duration:
                segments.append(self.video.subclip(start, end))
                current_duration += segment_duration
            else:
                break
                
        if not segments:
            # If no segments selected, take the first target_duration seconds
            return self.video.subclip(0, target_duration)
            
        # Combine segments with transitions
        final_clip = segments[0]
        for segment in segments[1:]:
            final_clip = self._create_transition(final_clip, segment)
            
        return final_clip
    
    def convert_to_vertical(self, target_width: int = 1080, target_height: int = 1920) -> VideoFileClip:
        """
        Convert video to vertical format (9:16 aspect ratio).
        
        Args:
            target_width (int): Target width in pixels
            target_height (int): Target height in pixels
            
        Returns:
            VideoFileClip: Vertically formatted video clip
        """
        # Calculate scaling factor to fit height
        scale_factor = target_height / self.video.h
        
        # Resize video
        resized = self.video.resize(width=int(self.video.w * scale_factor))
        
        # Create black background
        background = VideoFileClip(
            np.zeros((target_height, target_width, 3), dtype=np.uint8),
            duration=self.video.duration
        )
        
        # Center the video on the background
        x_offset = (target_width - resized.w) // 2
        y_offset = 0
        
        # Composite the video onto the background
        final = background.set_duration(self.video.duration)
        final = final.set_position((x_offset, y_offset))
        final = final.set_clip(resized)
        
        return final
    
    def _calculate_lufs(self, audio_segment: AudioSegment) -> float:
        """
        Calculate the integrated LUFS of an audio segment.
        
        Args:
            audio_segment (AudioSegment): Audio segment to analyze
            
        Returns:
            float: Integrated LUFS value
        """
        # Convert to numpy array
        samples = np.array(audio_segment.get_array_of_samples())
        
        # Calculate RMS energy
        rms = np.sqrt(np.mean(samples**2))
        
        # Convert to LUFS (approximate)
        # LUFS = -23 - 20 * log10(rms / 32768)
        lufs = -23 - 20 * np.log10(rms / 32768)
        
        return lufs
    
    def _normalize_lufs(self, audio_segment: AudioSegment, target_lufs: float) -> AudioSegment:
        """
        Normalize audio to target LUFS level.
        
        Args:
            audio_segment (AudioSegment): Audio segment to normalize
            target_lufs (float): Target LUFS level
            
        Returns:
            AudioSegment: Normalized audio segment
        """
        current_lufs = self._calculate_lufs(audio_segment)
        gain_db = target_lufs - current_lufs
        
        return audio_segment + gain_db
    
    def optimize_audio(self, target_lufs: float = -14.0) -> AudioFileClip:
        """
        Optimize audio levels for mobile playback.
        
        Args:
            target_lufs (float): Target LUFS level
            
        Returns:
            AudioFileClip: Optimized audio clip
        """
        if self.audio is None:
            return None
            
        # Convert to pydub AudioSegment for processing
        audio_segment = AudioSegment.from_file(self.input_path)
        
        # Normalize to target LUFS
        normalized_audio = self._normalize_lufs(audio_segment, target_lufs)
        
        # Save temporary file and load back as AudioFileClip
        temp_path = "temp_audio.wav"
        normalized_audio.export(temp_path, format="wav")
        optimized_audio = AudioFileClip(temp_path)
        os.remove(temp_path)
        
        return optimized_audio
    
    def _analyze_motion(self) -> List[Tuple[float, float, float, float]]:
        """
        Analyze video motion to determine optimal zoom/pan points.
        
        Returns:
            List[Tuple[float, float, float, float]]: List of (x, y, scale, timestamp) tuples
        """
        motion_points = []
        prev_frame = None
        
        for t in np.arange(0, self.video.duration, 0.5):
            frame = self.video.get_frame(t)
            if prev_frame is not None:
                # Calculate optical flow
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                
                # Calculate motion magnitude and direction
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                direction = np.arctan2(flow[..., 1], flow[..., 0])
                
                # Find center of motion
                if np.max(magnitude) > 1.0:
                    y, x = np.unravel_index(np.argmax(magnitude), magnitude.shape)
                    motion_points.append((x, y, 1.0, t))
            
            prev_frame = frame
            
        return motion_points
    
    def _apply_zoom_pan(self, clip: VideoFileClip) -> VideoFileClip:
        """
        Apply zoom and pan effects based on motion analysis.
        
        Args:
            clip (VideoFileClip): Input video clip
            
        Returns:
            VideoFileClip: Clip with zoom and pan effects
        """
        motion_points = self._analyze_motion()
        
        if not motion_points:
            return clip
            
        def zoom_pan(t):
            # Find closest motion point
            closest_point = min(motion_points, key=lambda p: abs(p[3] - t))
            x, y, scale, _ = closest_point
            
            # Calculate zoom and pan parameters
            zoom_factor = 1.0 + 0.2 * scale
            pan_x = (x - clip.w/2) / clip.w
            pan_y = (y - clip.h/2) / clip.h
            
            return zoom_factor, pan_x, pan_y
        
        # Apply zoom and pan effects
        return clip.fl(zoom_pan)
    
    def process_clip(self, target_duration: float = 30.0) -> Tuple[str, float]:
        """
        Process the video clip with all optimizations.
        
        Args:
            target_duration (float): Target duration in seconds
            
        Returns:
            Tuple[str, float]: Path to processed clip and final duration
        """
        try:
            # Optimize duration
            duration_optimized = self.optimize_duration(target_duration)
            
            # Apply zoom and pan effects
            zoom_panned = self._apply_zoom_pan(duration_optimized)
            
            # Convert to vertical format
            vertical_format = self.convert_to_vertical()
            
            # Optimize audio
            optimized_audio = self.optimize_audio()
            
            # Combine video and audio
            final_clip = vertical_format.set_audio(optimized_audio)
            
            # Export the final clip
            output_path = f"processed_{os.path.basename(self.input_path)}"
            final_clip.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile="temp-audio.m4a",
                remove_temp=True
            )
            
            return output_path, final_clip.duration
            
        except Exception as e:
            raise ProcessingError(f"Error processing clip: {str(e)}")
        finally:
            # Clean up temporary files
            if os.path.exists("temp_audio.wav"):
                os.remove("temp_audio.wav")
            if os.path.exists("temp-audio.m4a"):
                os.remove("temp-audio.m4a") 