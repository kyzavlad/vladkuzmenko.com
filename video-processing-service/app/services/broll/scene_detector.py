"""
Scene Detector module for B-Roll Insertion Engine.

This module handles the detection of scene transitions, changes in visual
composition, and identifies ideal insertion points for B-Roll footage.
"""

import os
import subprocess
import json
import logging
import asyncio
import tempfile
import math
import shutil
from typing import Dict, List, Any, Optional, Tuple, Union
import random

# Optional imports for advanced detection
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV not available. Using basic scene detection.")

logger = logging.getLogger(__name__)

class SceneDetector:
    """
    Detector for scene transitions and optimal B-Roll insertion points.
    
    This class analyzes video content to identify scene transitions, changes in
    visual composition, and optimal points for inserting B-Roll footage.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the SceneDetector.
        
        Args:
            config: Configuration options for scene detection
        """
        self.config = config or {}
        
        # Configure scene detection parameters
        self.min_scene_length = self.config.get('min_scene_length', 1.5)  # seconds
        self.max_scene_length = self.config.get('max_scene_length', 60.0)  # seconds
        self.detection_threshold = self.config.get('detection_threshold', 30.0)
        self.sample_rate = self.config.get('sample_rate', 1)  # 1 frame per second
        
        # Configure paths
        self.ffmpeg_path = self.config.get('ffmpeg_path', 'ffmpeg')
        self.ffprobe_path = self.config.get('ffprobe_path', 'ffprobe')
        
        # Configure temp directory
        self.temp_dir = self.config.get('temp_dir', tempfile.gettempdir())
        
        # Advanced options
        self.use_face_detection = self.config.get('use_face_detection', True) and CV2_AVAILABLE
        self.use_motion_detection = self.config.get('use_motion_detection', True) and CV2_AVAILABLE
        self.use_audio_analysis = self.config.get('use_audio_analysis', True)
        
        # Initialize face detector if available
        self.face_detector = None
        if self.use_face_detection and CV2_AVAILABLE:
            # Try to load face detector
            try:
                # Try to use CUDA if available
                face_cascade_path = self.config.get(
                    'face_cascade_path', 
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                self.face_detector = cv2.CascadeClassifier(face_cascade_path)
                logger.info("Face detector loaded successfully")
            except Exception as e:
                logger.error(f"Error loading face detector: {str(e)}")
                self.use_face_detection = False
    
    async def detect_scenes(self, video_path: str) -> Dict[str, Any]:
        """
        Detect scene transitions in a video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dict containing scene data
        """
        # Get video info
        video_info = await self._get_video_info(video_path)
        
        # Extract frames for analysis
        frames_dir = os.path.join(self.temp_dir, f"frames_{os.path.basename(video_path).split('.')[0]}")
        os.makedirs(frames_dir, exist_ok=True)
        
        frame_rate = video_info.get('frame_rate', 24)
        duration = video_info.get('duration', 0)
        
        # Extract frames for scene analysis
        frames = await self._extract_frames(
            video_path, 
            frames_dir, 
            sample_rate=self.sample_rate, 
            max_frames=min(300, int(duration * self.sample_rate))
        )
        
        # Detect scene transitions
        scenes = []
        current_scene_start = 0
        current_frame_index = 0
        
        # If we have extracted frames, analyze them for scene changes
        if frames:
            prev_frame = frames[0]
            for i, frame in enumerate(frames[1:], 1):
                frame_time = i / self.sample_rate
                
                # Detect if this is a scene change
                if await self._is_scene_change(prev_frame, frame):
                    # If the scene is long enough, add it
                    scene_duration = frame_time - current_scene_start
                    if scene_duration >= self.min_scene_length:
                        scenes.append({
                            'start': current_scene_start,
                            'end': frame_time,
                            'duration': scene_duration,
                            'start_frame': current_frame_index,
                            'end_frame': i
                        })
                    
                    # Start a new scene
                    current_scene_start = frame_time
                    current_frame_index = i
                
                prev_frame = frame
            
            # Add the final scene if it's long enough
            if duration - current_scene_start >= self.min_scene_length:
                scenes.append({
                    'start': current_scene_start,
                    'end': duration,
                    'duration': duration - current_scene_start,
                    'start_frame': current_frame_index,
                    'end_frame': len(frames) - 1
                })
        else:
            # If frame extraction failed, create a single scene
            scenes.append({
                'start': 0,
                'end': duration,
                'duration': duration,
                'start_frame': 0,
                'end_frame': 0
            })
        
        # Clean up frames
        shutil.rmtree(frames_dir, ignore_errors=True)
        
        return {
            'scenes': scenes,
            'video_info': video_info,
            'frame_rate': frame_rate,
            'sample_rate': self.sample_rate
        }
    
    async def find_broll_insertion_points(
        self, 
        video_path: str,
        transcript: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Find optimal insertion points for B-Roll footage.
        
        Args:
            video_path: Path to the video file
            transcript: Optional transcript with timing information
            
        Returns:
            Dict containing insertion point data
        """
        scene_data = await self.detect_scenes(video_path)
        scenes = scene_data.get('scenes', [])
        
        # Analyze visual composition of scenes
        scene_composition = []
        for scene in scenes:
            composition = await self._analyze_scene_composition(
                video_path, 
                scene.get('start', 0), 
                scene.get('end', 0)
            )
            
            scene_composition.append({
                'start': scene.get('start', 0),
                'end': scene.get('end', 0),
                'composition': composition
            })
        
        # Identify potential insertion points
        insertion_points = []
        
        # First, identify points from scene transitions
        for i, scene in enumerate(scenes):
            scene_start = scene.get('start', 0)
            scene_end = scene.get('end', 0)
            
            # Add the beginning of each scene (except the first)
            if i > 0:
                insertion_points.append({
                    'timestamp': scene_start,
                    'type': 'scene_transition',
                    'score': 8.0,
                    'recommended_duration': min(5.0, scene.get('duration', 5.0) * 0.7)
                })
            
            # Add midpoints for longer scenes
            if scene.get('duration', 0) > 10.0:
                midpoint = scene_start + (scene_end - scene_start) / 2
                insertion_points.append({
                    'timestamp': midpoint,
                    'type': 'scene_midpoint',
                    'score': 6.0,
                    'recommended_duration': min(4.0, scene.get('duration', 10.0) * 0.4)
                })
        
        # If transcript is provided, add insertion points based on content
        if transcript:
            # Process transcript segments
            segments = transcript.get('segments', [])
            
            for segment in segments:
                segment_start = segment.get('start', 0)
                segment_end = segment.get('end', 0)
                segment_text = segment.get('text', '')
                
                # Skip very short segments
                if segment_end - segment_start < 2.0 or not segment_text:
                    continue
                
                # Check if this segment overlaps with a scene transition
                overlaps_transition = False
                for point in insertion_points:
                    if (point.get('type') == 'scene_transition' and 
                        abs(point.get('timestamp', 0) - segment_start) < 2.0):
                        overlaps_transition = True
                        break
                
                if not overlaps_transition:
                    # Add new insertion point at the beginning of the segment
                    insertion_points.append({
                        'timestamp': segment_start,
                        'type': 'segment_start',
                        'score': 7.0,
                        'segment_start': segment_start,
                        'segment_end': segment_end,
                        'segment_text': segment_text,
                        'recommended_duration': min(4.0, (segment_end - segment_start) * 0.7)
                    })
        
        # Detect audio transitions (pauses, shifts in volume)
        audio_transitions = await self._detect_audio_transitions(video_path)
        
        # Add insertion points based on audio transitions
        for transition in audio_transitions:
            timestamp = transition.get('timestamp', 0)
            
            # Check if this is near an existing insertion point
            is_near_existing = False
            for point in insertion_points:
                if abs(point.get('timestamp', 0) - timestamp) < 2.0:
                    is_near_existing = True
                    # If this is an audio pause, increase the score of the existing point
                    if transition.get('type') == 'pause':
                        point['score'] = max(point.get('score', 0), 7.5)
                    break
            
            if not is_near_existing:
                insertion_points.append({
                    'timestamp': timestamp,
                    'type': transition.get('type', 'audio_transition'),
                    'score': 6.5 if transition.get('type') == 'pause' else 5.5,
                    'recommended_duration': 3.0
                })
        
        # Sort insertion points by timestamp
        insertion_points.sort(key=lambda x: x.get('timestamp', 0))
        
        # Remove points that are too close together
        filtered_points = []
        min_spacing = 5.0  # Minimum time between insertion points (seconds)
        
        for point in insertion_points:
            if not filtered_points or point.get('timestamp', 0) - filtered_points[-1].get('timestamp', 0) >= min_spacing:
                # If transcript is available, associate segment text with this point
                if transcript and 'segment_text' not in point:
                    # Find the segment that contains this timestamp
                    for segment in transcript.get('segments', []):
                        if segment.get('start', 0) <= point.get('timestamp', 0) <= segment.get('end', 0):
                            point['segment_start'] = segment.get('start', 0)
                            point['segment_end'] = segment.get('end', 0)
                            point['segment_text'] = segment.get('text', '')
                            break
                
                filtered_points.append(point)
        
        return {
            'scenes': scene_data,
            'insertion_points': filtered_points,
            'composition': scene_composition
        }
    
    async def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get information about a video file using FFprobe.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dict containing video metadata
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return {}
        
        try:
            cmd = [
                self.ffprobe_path,
                '-v', 'error',
                '-show_entries', 'format=duration,size,bit_rate:stream=width,height,r_frame_rate,codec_name',
                '-of', 'json',
                video_path
            ]
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                logger.error(f"Error getting video info: {stderr.decode()}")
                return {}
            
            data = json.loads(stdout.decode())
            format_data = data.get('format', {})
            streams = data.get('streams', [])
            
            # Find video stream
            video_stream = None
            for stream in streams:
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            if not video_stream:
                for stream in streams:
                    if 'width' in stream and 'height' in stream:
                        video_stream = stream
                        break
            
            # Extract relevant information
            video_info = {
                'duration': float(format_data.get('duration', 0)),
                'size': int(format_data.get('size', 0)),
                'bit_rate': int(format_data.get('bit_rate', 0))
            }
            
            if video_stream:
                # Parse frame rate (could be a fraction like "24000/1001")
                frame_rate_str = video_stream.get('r_frame_rate', '24/1')
                if '/' in frame_rate_str:
                    num, den = map(int, frame_rate_str.split('/'))
                    frame_rate = num / den if den != 0 else 24
                else:
                    frame_rate = float(frame_rate_str)
                
                video_info.update({
                    'width': int(video_stream.get('width', 0)),
                    'height': int(video_stream.get('height', 0)),
                    'frame_rate': frame_rate,
                    'codec': video_stream.get('codec_name', '')
                })
            
            return video_info
            
        except Exception as e:
            logger.error(f"Error getting video info: {str(e)}")
            return {}
    
    async def _extract_frames(
        self, 
        video_path: str, 
        output_dir: str, 
        sample_rate: float = 1.0,
        max_frames: int = 300
    ) -> List[str]:
        """
        Extract frames from a video for analysis.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save extracted frames
            sample_rate: Number of frames to extract per second
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of paths to extracted frames
        """
        try:
            # Get video duration
            video_info = await self._get_video_info(video_path)
            duration = video_info.get('duration', 0)
            
            if duration == 0:
                logger.error("Unable to determine video duration")
                return []
            
            # Calculate frame interval
            interval = 1.0 / sample_rate
            num_frames = min(max_frames, int(duration * sample_rate))
            
            # Build FFmpeg command
            cmd = [
                self.ffmpeg_path,
                '-i', video_path,
                '-vf', f'fps={sample_rate}',
                '-q:v', '2',
                '-frames:v', str(num_frames),
                os.path.join(output_dir, 'frame_%04d.jpg')
            ]
            
            # Execute FFmpeg
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Error extracting frames: {stderr.decode()}")
                return []
            
            # Get list of extracted frames
            frames = []
            for i in range(1, num_frames + 1):
                frame_path = os.path.join(output_dir, f'frame_{i:04d}.jpg')
                if os.path.exists(frame_path):
                    frames.append(frame_path)
            
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            return []
    
    async def _is_scene_change(self, frame1_path: str, frame2_path: str) -> bool:
        """
        Determine if there is a scene change between two frames.
        
        Args:
            frame1_path: Path to the first frame
            frame2_path: Path to the second frame
            
        Returns:
            True if there is a scene change, False otherwise
        """
        if not CV2_AVAILABLE:
            # Fallback - use file size difference as very basic heuristic
            try:
                size1 = os.path.getsize(frame1_path)
                size2 = os.path.getsize(frame2_path)
                difference = abs(size1 - size2) / max(size1, size2) * 100
                return difference > self.detection_threshold
            except:
                return False
        
        try:
            # Read frames
            frame1 = cv2.imread(frame1_path)
            frame2 = cv2.imread(frame2_path)
            
            if frame1 is None or frame2 is None:
                return False
            
            # Convert to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Calculate histogram
            hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
            
            # Normalize histograms
            cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
            
            # Compare histograms
            difference = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA) * 100
            
            # Calculate structural similarity
            try:
                from skimage.metrics import structural_similarity as ssim
                gray1_resized = cv2.resize(gray1, (320, 240))
                gray2_resized = cv2.resize(gray2, (320, 240))
                similarity = ssim(gray1_resized, gray2_resized)
                
                # Combine metrics
                combined_diff = difference + (1 - similarity) * 50
                return combined_diff > self.detection_threshold
            except ImportError:
                # Fall back to just histogram difference
                return difference > self.detection_threshold
            
        except Exception as e:
            logger.error(f"Error comparing frames: {str(e)}")
            return False
    
    async def _analyze_scene_composition(
        self, 
        video_path: str, 
        start_time: float, 
        end_time: float
    ) -> Dict[str, Any]:
        """
        Analyze the visual composition of a scene.
        
        Args:
            video_path: Path to the video file
            start_time: Start time of the scene (seconds)
            end_time: End time of the scene (seconds)
            
        Returns:
            Dict containing composition analysis
        """
        composition = {
            'faces': [],
            'complexity': 0.0,
            'motion': 0.0,
            'dominant_colors': [],
            'empty_areas': []
        }
        
        if not CV2_AVAILABLE:
            # Without OpenCV, return basic composition
            return composition
        
        try:
            # Extract a few frames from the scene
            frames_dir = os.path.join(self.temp_dir, f"scene_frames_{os.path.basename(video_path).split('.')[0]}_{int(start_time)}")
            os.makedirs(frames_dir, exist_ok=True)
            
            # Sample up to 5 frames from the scene
            scene_duration = end_time - start_time
            num_samples = min(5, max(2, int(scene_duration)))
            sample_times = [start_time + (i * scene_duration / (num_samples - 1)) for i in range(num_samples)]
            
            frames = []
            for i, time in enumerate(sample_times):
                frame_path = os.path.join(frames_dir, f'scene_frame_{i:02d}.jpg')
                
                # Extract frame using FFmpeg
                cmd = [
                    self.ffmpeg_path,
                    '-ss', str(time),
                    '-i', video_path,
                    '-frames:v', '1',
                    '-q:v', '2',
                    frame_path
                ]
                
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                await process.communicate()
                
                if os.path.exists(frame_path):
                    frames.append(frame_path)
            
            # Analyze frames
            face_detections = []
            complexity_scores = []
            dominant_colors_list = []
            empty_areas_list = []
            
            for frame_path in frames:
                # Read frame
                frame = cv2.imread(frame_path)
                
                if frame is None:
                    continue
                
                # Detect faces
                if self.use_face_detection and self.face_detector is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_detector.detectMultiScale(
                        gray, 
                        scaleFactor=1.1, 
                        minNeighbors=5,
                        minSize=(30, 30)
                    )
                    
                    # Convert face coordinates to relative positions
                    height, width = frame.shape[:2]
                    
                    for (x, y, w, h) in faces:
                        rel_x = x / width
                        rel_y = y / height
                        rel_w = w / width
                        rel_h = h / height
                        
                        face_detections.append({
                            'x': rel_x,
                            'y': rel_y,
                            'width': rel_w,
                            'height': rel_h,
                            'area': rel_w * rel_h
                        })
                
                # Calculate image complexity
                try:
                    # Compute edges using Sobel
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                    
                    # Calculate gradient magnitude
                    magnitude = np.sqrt(sobelx**2 + sobely**2)
                    
                    # Normalize and get mean complexity
                    complexity = np.mean(magnitude) / 255.0
                    complexity_scores.append(complexity)
                except Exception as e:
                    logger.error(f"Error calculating image complexity: {str(e)}")
                
                # Extract dominant colors
                try:
                    # Resize to speed up processing
                    small_frame = cv2.resize(frame, (100, 100))
                    pixels = small_frame.reshape(-1, 3)
                    
                    # K-means clustering
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=5, random_state=42)
                    kmeans.fit(pixels)
                    
                    # Get dominant colors
                    colors = kmeans.cluster_centers_.astype(int)
                    counts = np.bincount(kmeans.labels_)
                    
                    # Sort by frequency
                    dominant = []
                    for i, count in enumerate(counts):
                        color = colors[i].tolist()
                        dominant.append({
                            'color': color,
                            'hex': f'#{color[2]:02x}{color[1]:02x}{color[0]:02x}',
                            'percentage': count / len(pixels)
                        })
                    
                    dominant_colors_list.append(dominant)
                except Exception as e:
                    # Skip color detection if sklearn is not available
                    pass
                
                # Find empty areas (regions with low texture/variance)
                try:
                    # Divide the image into a 3x3 grid
                    height, width = frame.shape[:2]
                    cell_h = height // 3
                    cell_w = width // 3
                    
                    empty_cells = []
                    
                    for y in range(3):
                        for x in range(3):
                            # Get cell coordinates
                            x1 = x * cell_w
                            y1 = y * cell_h
                            x2 = min((x + 1) * cell_w, width)
                            y2 = min((y + 1) * cell_h, height)
                            
                            # Extract cell
                            cell = gray[y1:y2, x1:x2]
                            
                            # Calculate variance (texture)
                            variance = np.var(cell)
                            
                            # If variance is low, consider it an empty area
                            if variance < 500:  # Threshold for "emptiness"
                                empty_cells.append({
                                    'x': x / 3,
                                    'y': y / 3,
                                    'width': 1/3,
                                    'height': 1/3,
                                    'emptiness': 1.0 - (variance / 500)
                                })
                    
                    empty_areas_list.append(empty_cells)
                except Exception as e:
                    logger.error(f"Error finding empty areas: {str(e)}")
            
            # Clean up
            shutil.rmtree(frames_dir, ignore_errors=True)
            
            # Aggregate results
            if face_detections:
                composition['faces'] = face_detections
            
            if complexity_scores:
                composition['complexity'] = sum(complexity_scores) / len(complexity_scores)
            
            if dominant_colors_list:
                # Combine dominant colors across frames
                all_colors = {}
                
                for colors in dominant_colors_list:
                    for color_info in colors:
                        hex_color = color_info.get('hex', '')
                        if hex_color in all_colors:
                            all_colors[hex_color]['percentage'] += color_info.get('percentage', 0)
                        else:
                            all_colors[hex_color] = {
                                'color': color_info.get('color', []),
                                'hex': hex_color,
                                'percentage': color_info.get('percentage', 0)
                            }
                
                # Normalize percentages
                for hex_color in all_colors:
                    all_colors[hex_color]['percentage'] /= len(dominant_colors_list)
                
                # Sort by percentage
                composition['dominant_colors'] = sorted(
                    all_colors.values(), 
                    key=lambda x: x.get('percentage', 0),
                    reverse=True
                )
            
            if empty_areas_list:
                # Find consistent empty areas across frames
                grid = np.zeros((3, 3))
                for areas in empty_areas_list:
                    for area in areas:
                        x = int(area.get('x', 0) * 3)
                        y = int(area.get('y', 0) * 3)
                        emptiness = area.get('emptiness', 0)
                        grid[y, x] += emptiness
                
                # Normalize
                grid = grid / len(empty_areas_list)
                
                # Find cells with high emptiness
                for y in range(3):
                    for x in range(3):
                        if grid[y, x] > 0.5:  # Threshold for consistent emptiness
                            composition['empty_areas'].append({
                                'x': x / 3,
                                'y': y / 3,
                                'width': 1/3,
                                'height': 1/3,
                                'emptiness': grid[y, x]
                            })
            
            return composition
            
        except Exception as e:
            logger.error(f"Error analyzing scene composition: {str(e)}")
            return composition
    
    async def _detect_audio_transitions(
        self, 
        video_path: str
    ) -> List[Dict[str, Any]]:
        """
        Detect audio transitions (pauses, volume changes) in a video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of audio transition points
        """
        if not self.use_audio_analysis:
            return []
        
        try:
            # Extract audio using FFmpeg
            audio_file = os.path.join(self.temp_dir, f"audio_{os.path.basename(video_path).split('.')[0]}.wav")
            
            cmd = [
                self.ffmpeg_path,
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM format
                '-ar', '16000',  # 16 kHz sampling rate
                '-ac', '1',  # Mono
                audio_file
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await process.communicate()
            
            if not os.path.exists(audio_file):
                logger.error("Failed to extract audio")
                return []
            
            # Analyze audio for silence
            silence_file = os.path.join(self.temp_dir, f"silence_{os.path.basename(video_path).split('.')[0]}.txt")
            
            cmd = [
                self.ffmpeg_path,
                '-i', audio_file,
                '-af', 'silencedetect=noise=-30dB:d=0.5',  # Detect silence (adjust parameters as needed)
                '-f', 'null',
                '-'
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            # Parse silence detection output
            silence_output = stderr.decode()
            transitions = []
            
            # Extract silence start/end times
            import re
            silence_starts = re.findall(r'silence_start: (\d+\.?\d*)', silence_output)
            silence_ends = re.findall(r'silence_end: (\d+\.?\d*)', silence_output)
            
            # Add pauses as transitions
            for start, end in zip(silence_starts, silence_ends):
                start_time = float(start)
                end_time = float(end)
                
                # Only consider actual pauses (not just brief moments)
                if end_time - start_time >= 0.5:
                    # Add the end of silence as a good transition point
                    transitions.append({
                        'timestamp': end_time,
                        'type': 'pause',
                        'duration': end_time - start_time
                    })
            
            # Clean up
            os.remove(audio_file)
            
            return transitions
            
        except Exception as e:
            logger.error(f"Error detecting audio transitions: {str(e)}")
            return []


def _parse_framerate(framerate_str: str) -> float:
    """
    Parse a framerate string (e.g., '30/1', '24000/1001').
    
    Args:
        framerate_str: Framerate string in the format 'num/den'
        
    Returns:
        Framerate as a float
    """
    try:
        if '/' in framerate_str:
            num, den = map(int, framerate_str.split('/'))
            return num / den
        else:
            return float(framerate_str)
    except (ValueError, ZeroDivisionError):
        return 30.0  # Default to 30 fps 