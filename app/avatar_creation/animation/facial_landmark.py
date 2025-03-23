import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any
import time

from app.avatar_creation.face_modeling.utils import get_device, ensure_directory

class FacialLandmarkTracker:
    """
    Class for detecting and tracking 68-point facial landmarks in images and videos.
    Supports smooth tracking with temporal consistency.
    """
    
    def __init__(self, 
                model_path: Optional[str] = None,
                use_gpu: bool = True,
                temporal_smoothing: float = 0.5,
                detection_confidence: float = 0.7):
        """
        Initialize the facial landmark tracker.
        
        Args:
            model_path: Path to pre-trained landmark detector model
            use_gpu: Whether to use GPU acceleration
            temporal_smoothing: Smoothing factor for temporal consistency
            detection_confidence: Confidence threshold for face detection
        """
        self.device = get_device() if use_gpu else torch.device("cpu")
        self.model_path = model_path
        self.temporal_smoothing = temporal_smoothing
        self.detection_confidence = detection_confidence
        
        # Initialize detector and predictor
        self.face_detector = self._initialize_face_detector()
        self.landmark_predictor = self._initialize_landmark_predictor()
        
        # Previous landmarks for temporal consistency
        self.prev_landmarks = None
        self.landmark_history = []
        self.max_history_length = 30  # frames
        
        # For tracking statistics
        self.frame_times = []
        self.last_detection_time = time.time()
        
        print(f"Facial Landmark Tracker initialized")
        print(f"  - Device: {self.device}")
        print(f"  - Face detector: {'Loaded' if self.face_detector else 'None'}")
        print(f"  - Landmark predictor: {'Loaded' if self.landmark_predictor else 'None'}")
    
    def _initialize_face_detector(self) -> Any:
        """
        Initialize the face detector.
        
        Returns:
            Face detector model
        """
        try:
            # Try to use dlib's face detector
            try:
                import dlib
                detector = dlib.get_frontal_face_detector()
                print("Using dlib's frontal face detector")
                return detector
            except ImportError:
                print("dlib not available, using OpenCV's face detector")
            
            # Fallback to OpenCV's face detector
            face_cascade_path = os.path.join(cv2.__path__[0], 'data', 
                                           'haarcascade_frontalface_default.xml')
            
            if not os.path.exists(face_cascade_path):
                # Try alternate path for OpenCV data
                alt_path = "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
                if os.path.exists(alt_path):
                    face_cascade_path = alt_path
                else:
                    print(f"Haar cascade file not found at {face_cascade_path}")
                    return None
            
            face_cascade = cv2.CascadeClassifier(face_cascade_path)
            return face_cascade
            
        except Exception as e:
            print(f"Error initializing face detector: {e}")
            return None
    
    def _initialize_landmark_predictor(self) -> Any:
        """
        Initialize the landmark predictor.
        
        Returns:
            Landmark predictor model
        """
        try:
            # Check if model path is provided
            if self.model_path and os.path.exists(self.model_path):
                # Try to load dlib's facial landmark predictor
                try:
                    import dlib
                    predictor = dlib.shape_predictor(self.model_path)
                    print(f"Using dlib's landmark predictor from: {self.model_path}")
                    return predictor
                except ImportError:
                    print("dlib not available, using placeholder landmark predictor")
            else:
                print("No landmark predictor model found. Using placeholder implementation.")
            
            # Create a simple placeholder predictor
            class PlaceholderLandmarkPredictor:
                def __call__(self, image, face_rect):
                    """Predict facial landmarks using a simple geometric approximation"""
                    # Extract face dimensions
                    if isinstance(face_rect, dlib.rectangle):
                        x, y = face_rect.left(), face_rect.top()
                        w, h = face_rect.width(), face_rect.height()
                    else:
                        x, y, w, h = face_rect
                    
                    # Generate 68 landmark points approximation
                    landmarks = np.zeros((68, 2), dtype=np.float32)
                    
                    # Different regions of the face
                    regions = {
                        'jaw': (0, 17),
                        'right_eyebrow': (17, 22),
                        'left_eyebrow': (22, 27),
                        'nose_bridge': (27, 31),
                        'nose_tip': (31, 36),
                        'right_eye': (36, 42),
                        'left_eye': (42, 48),
                        'outer_lips': (48, 60),
                        'inner_lips': (60, 68)
                    }
                    
                    # Generate points for each region
                    face_center_x = x + w // 2
                    face_center_y = y + h // 2
                    
                    for region_name, (start_idx, end_idx) in regions.items():
                        num_points = end_idx - start_idx
                        
                        if region_name == 'jaw':
                            # Jawline: from ear to ear
                            x_coords = np.linspace(x + w // 8, x + w - w // 8, num_points)
                            y_coords = np.linspace(y + h * 3 // 4, y + h * 3 // 4, num_points)
                            y_coords = y_coords + np.linspace(-h // 10, h // 10, num_points)
                            
                        elif 'eyebrow' in region_name:
                            # Eyebrows
                            offset_x = w // 4 if 'right' in region_name else 3 * w // 4
                            x_coords = np.linspace(x + offset_x - w // 10, x + offset_x + w // 10, num_points)
                            y_coords = np.ones(num_points) * (y + h // 4)
                            
                        elif 'eye' in region_name:
                            # Eyes
                            offset_x = w // 3 if 'right' in region_name else 2 * w // 3
                            theta = np.linspace(0, 2*np.pi, num_points+1)[:-1]
                            eye_w, eye_h = w // 12, h // 18
                            x_coords = x + offset_x + eye_w * np.cos(theta)
                            y_coords = y + h * 3 // 8 + eye_h * np.sin(theta)
                            
                        elif 'nose' in region_name:
                            # Nose
                            if 'bridge' in region_name:
                                # Nose bridge
                                x_coords = np.ones(num_points) * face_center_x
                                y_coords = np.linspace(y + h // 3, y + h // 2, num_points)
                            else:
                                # Nose tip
                                x_start = face_center_x - w // 8
                                x_end = face_center_x + w // 8
                                x_coords = np.linspace(x_start, x_end, num_points)
                                y_coords = np.ones(num_points) * (y + h * 5 // 8)
                        
                        elif 'lips' in region_name:
                            # Lips
                            is_inner = 'inner' in region_name
                            scale = 0.7 if is_inner else 1.0
                            
                            theta = np.linspace(0, 2*np.pi, num_points+1)[:-1]
                            lip_w = w // 4 * scale
                            lip_h = h // 12 * scale
                            x_coords = face_center_x + lip_w * np.cos(theta)
                            y_coords = y + h * 3 // 4 + lip_h * np.sin(theta)
                        
                        # Assign to landmarks array
                        landmarks[start_idx:end_idx, 0] = x_coords
                        landmarks[start_idx:end_idx, 1] = y_coords
                    
                    # Add small random noise for more realism
                    landmarks += np.random.normal(0, 2, landmarks.shape)
                    
                    # Create a shape object similar to dlib's full_object_detection
                    class Shape:
                        def __init__(self, points):
                            self.points = points
                        
                        def part(self, idx):
                            class Point:
                                def __init__(self, x, y):
                                    self.x = x
                                    self.y = y
                            return Point(self.points[idx][0], self.points[idx][1])
                        
                        def parts(self):
                            return [self.part(i) for i in range(len(self.points))]
                    
                    return Shape(landmarks)
            
            return PlaceholderLandmarkPredictor()
            
        except Exception as e:
            print(f"Error initializing landmark predictor: {e}")
            return None
    
    def detect_face(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image
            
        Returns:
            List of face rectangles as (x, y, w, h)
        """
        if self.face_detector is None:
            print("No face detector available")
            # Return a centered face covering most of the image as fallback
            h, w = image.shape[:2]
            face_w, face_h = int(w * 0.7), int(h * 0.7)
            x, y = (w - face_w) // 2, (h - face_h) // 2
            return [(x, y, face_w, face_h)]
        
        # Convert to grayscale for detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        try:
            # Check which detector we have
            import inspect
            if 'dlib' in str(inspect.getmodule(self.face_detector.__class__)):
                # dlib detector
                dlib_rects = self.face_detector(gray, 1)
                return [(rect.left(), rect.top(), rect.width(), rect.height()) 
                        for rect in dlib_rects]
            else:
                # OpenCV detector
                faces = self.face_detector.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                return faces.tolist() if len(faces) > 0 else []
                
        except Exception as e:
            print(f"Error in face detection: {e}")
            # Return a centered face as fallback
            h, w = image.shape[:2]
            face_w, face_h = int(w * 0.7), int(h * 0.7)
            x, y = (w - face_w) // 2, (h - face_h) // 2
            return [(x, y, face_w, face_h)]
    
    def detect_landmarks(self, image: np.ndarray, face_rect: Union[Tuple[int, int, int, int], Any]) -> np.ndarray:
        """
        Detect facial landmarks for a face.
        
        Args:
            image: Input image
            face_rect: Face rectangle as (x, y, w, h) or dlib rectangle
            
        Returns:
            Array of 68 landmark points as (x, y) coordinates
        """
        if self.landmark_predictor is None:
            print("No landmark predictor available")
            # Return placeholder landmarks
            if isinstance(face_rect, tuple):
                x, y, w, h = face_rect
            else:
                # Assume dlib rectangle
                x, y = face_rect.left(), face_rect.top()
                w, h = face_rect.width(), face_rect.height()
                
            # Generate a grid of points as placeholder
            landmarks = np.zeros((68, 2), dtype=np.float32)
            for i in range(68):
                row, col = i // 8, i % 8
                landmarks[i, 0] = x + col * w / 8
                landmarks[i, 1] = y + row * h / 8
                
            return landmarks
        
        try:
            # Convert to grayscale for prediction
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Handle different rectangle formats
            if isinstance(face_rect, tuple):
                x, y, w, h = face_rect
                # Convert to dlib rectangle if needed
                try:
                    import dlib
                    dlib_rect = dlib.rectangle(x, y, x+w, y+h)
                except ImportError:
                    dlib_rect = face_rect
            else:
                dlib_rect = face_rect
            
            # Predict landmarks
            shape = self.landmark_predictor(gray, dlib_rect)
            
            # Convert to numpy array
            landmarks = np.zeros((68, 2), dtype=np.float32)
            for i in range(68):
                try:
                    landmarks[i, 0] = shape.part(i).x
                    landmarks[i, 1] = shape.part(i).y
                except:
                    # In case the predictor doesn't return all points
                    pass
            
            return landmarks
            
        except Exception as e:
            print(f"Error in landmark detection: {e}")
            # Return placeholder landmarks
            if isinstance(face_rect, tuple):
                x, y, w, h = face_rect
            else:
                # Try to extract coordinates from dlib rectangle
                try:
                    x, y = face_rect.left(), face_rect.top()
                    w, h = face_rect.width(), face_rect.height()
                except:
                    # Fallback to image dimensions
                    h, w = image.shape[:2]
                    x, y = 0, 0
                    
            # Generate placeholder landmarks based on face rectangle
            landmarks = np.zeros((68, 2), dtype=np.float32)
            
            # Different regions of the face
            regions = {
                'jaw': (0, 17),
                'right_eyebrow': (17, 22),
                'left_eyebrow': (22, 27),
                'nose_bridge': (27, 31),
                'nose_tip': (31, 36),
                'right_eye': (36, 42),
                'left_eye': (42, 48),
                'outer_lips': (48, 60),
                'inner_lips': (60, 68)
            }
            
            # Generate points for each region
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            
            for region_name, (start_idx, end_idx) in regions.items():
                num_points = end_idx - start_idx
                
                if region_name == 'jaw':
                    # Jawline: from ear to ear
                    x_coords = np.linspace(x + w // 8, x + w - w // 8, num_points)
                    y_coords = np.linspace(y + h * 3 // 4, y + h * 3 // 4, num_points)
                    y_coords = y_coords + np.linspace(-h // 10, h // 10, num_points)
                    
                elif 'eyebrow' in region_name:
                    # Eyebrows
                    offset_x = w // 4 if 'right' in region_name else 3 * w // 4
                    x_coords = np.linspace(x + offset_x - w // 10, x + offset_x + w // 10, num_points)
                    y_coords = np.ones(num_points) * (y + h // 4)
                    
                elif 'eye' in region_name:
                    # Eyes
                    offset_x = w // 3 if 'right' in region_name else 2 * w // 3
                    theta = np.linspace(0, 2*np.pi, num_points+1)[:-1]
                    eye_w, eye_h = w // 12, h // 18
                    x_coords = x + offset_x + eye_w * np.cos(theta)
                    y_coords = y + h * 3 // 8 + eye_h * np.sin(theta)
                    
                elif 'nose' in region_name:
                    # Nose
                    if 'bridge' in region_name:
                        # Nose bridge
                        x_coords = np.ones(num_points) * face_center_x
                        y_coords = np.linspace(y + h // 3, y + h // 2, num_points)
                    else:
                        # Nose tip
                        x_start = face_center_x - w // 8
                        x_end = face_center_x + w // 8
                        x_coords = np.linspace(x_start, x_end, num_points)
                        y_coords = np.ones(num_points) * (y + h * 5 // 8)
                
                elif 'lips' in region_name:
                    # Lips
                    is_inner = 'inner' in region_name
                    scale = 0.7 if is_inner else 1.0
                    
                    theta = np.linspace(0, 2*np.pi, num_points+1)[:-1]
                    lip_w = w // 4 * scale
                    lip_h = h // 12 * scale
                    x_coords = face_center_x + lip_w * np.cos(theta)
                    y_coords = y + h * 3 // 4 + lip_h * np.sin(theta)
                
                # Assign to landmarks array
                landmarks[start_idx:end_idx, 0] = x_coords
                landmarks[start_idx:end_idx, 1] = y_coords
            
            # Add small random noise for more realism
            landmarks += np.random.normal(0, 2, landmarks.shape)
                
            return landmarks
    
    def track_landmarks(self, image: np.ndarray) -> Dict:
        """
        Track facial landmarks in an image with temporal consistency.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with landmarks and face information
        """
        start_time = time.time()
        
        # Detect faces
        faces = self.detect_face(image)
        
        # No faces found
        if not faces:
            return {
                'success': False,
                'message': 'No faces detected',
                'landmarks': None,
                'face_rect': None,
                'processing_time': time.time() - start_time
            }
        
        # Use the largest face (or first face)
        face_rect = max(faces, key=lambda rect: rect[2] * rect[3])
        
        # Detect landmarks
        landmarks = self.detect_landmarks(image, face_rect)
        
        # Apply temporal smoothing
        if self.prev_landmarks is not None and self.temporal_smoothing > 0:
            landmarks = (1 - self.temporal_smoothing) * landmarks + \
                       self.temporal_smoothing * self.prev_landmarks
        
        # Update previous landmarks
        self.prev_landmarks = landmarks.copy()
        
        # Update landmark history
        self.landmark_history.append({
            'timestamp': time.time(),
            'landmarks': landmarks.copy()
        })
        
        # Trim history if needed
        if len(self.landmark_history) > self.max_history_length:
            self.landmark_history.pop(0)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        self.frame_times.append(processing_time)
        if len(self.frame_times) > 100:
            self.frame_times.pop(0)
        
        # Calculate average processing time
        avg_processing_time = sum(self.frame_times) / len(self.frame_times)
        
        return {
            'success': True,
            'landmarks': landmarks,
            'face_rect': face_rect,
            'processing_time': processing_time,
            'avg_processing_time': avg_processing_time
        }
    
    def track_landmarks_in_video(self, video_path: str, output_path: Optional[str] = None) -> Dict:
        """
        Track facial landmarks in a video.
        
        Args:
            video_path: Path to input video
            output_path: Path to save visualization video (optional)
            
        Returns:
            Dictionary with tracking results
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                'success': False,
                'message': f'Could not open video: {video_path}'
            }
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if output path is provided
        video_writer = None
        if output_path:
            ensure_directory(os.path.dirname(output_path))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Reset tracker state
        self.prev_landmarks = None
        self.landmark_history = []
        
        # Process each frame
        landmarks_data = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Track landmarks
            result = self.track_landmarks(frame)
            
            # Save landmarks data
            if result['success']:
                landmarks_data.append({
                    'frame_idx': frame_idx,
                    'timestamp': frame_idx / fps,
                    'landmarks': result['landmarks'].tolist(),
                    'face_rect': result['face_rect']
                })
                
                # Draw landmarks on frame for visualization
                if video_writer is not None:
                    vis_frame = frame.copy()
                    
                    # Draw face rectangle
                    x, y, w, h = result['face_rect']
                    cv2.rectangle(vis_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Draw landmarks
                    for i, (lx, ly) in enumerate(result['landmarks']):
                        cv2.circle(vis_frame, (int(lx), int(ly)), 2, (0, 0, 255), -1)
                        
                    # Add frame info
                    cv2.putText(vis_frame, f"Frame: {frame_idx}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Write frame
                    video_writer.write(vis_frame)
            
            frame_idx += 1
            
            # Print progress
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{frame_count} frames ({frame_idx/frame_count*100:.1f}%)")
        
        # Release resources
        cap.release()
        if video_writer:
            video_writer.release()
        
        return {
            'success': True,
            'total_frames': frame_idx,
            'landmark_frames': len(landmarks_data),
            'landmarks_data': landmarks_data,
            'visualization_path': output_path if output_path else None
        }
    
    def visualize_landmarks(self, image: np.ndarray, landmarks: np.ndarray,
                          face_rect: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        Visualize facial landmarks on an image.
        
        Args:
            image: Input image
            landmarks: Array of landmark points
            face_rect: Face rectangle as (x, y, w, h)
            
        Returns:
            Image with visualized landmarks
        """
        vis_image = image.copy()
        
        # Draw face rectangle if provided
        if face_rect:
            x, y, w, h = face_rect
            cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Define landmark regions and colors
        regions = {
            'jaw': (0, 17, (0, 255, 0)),      # Green
            'right_eyebrow': (17, 22, (255, 0, 0)),  # Blue
            'left_eyebrow': (22, 27, (255, 0, 0)),   # Blue
            'nose_bridge': (27, 31, (0, 0, 255)),    # Red
            'nose_tip': (31, 36, (0, 0, 255)),       # Red
            'right_eye': (36, 42, (255, 255, 0)),    # Cyan
            'left_eye': (42, 48, (255, 255, 0)),     # Cyan
            'outer_lips': (48, 60, (255, 0, 255)),   # Magenta
            'inner_lips': (60, 68, (255, 0, 255))    # Magenta
        }
        
        # Draw each region
        for region_name, (start_idx, end_idx, color) in regions.items():
            # Draw lines connecting points
            for i in range(start_idx, end_idx - 1):
                cv2.line(vis_image, 
                        (int(landmarks[i, 0]), int(landmarks[i, 1])),
                        (int(landmarks[i+1, 0]), int(landmarks[i+1, 1])),
                        color, 2)
                
            # Connect the last point to the first for closed regions
            if region_name in ['right_eye', 'left_eye', 'outer_lips', 'inner_lips']:
                cv2.line(vis_image,
                        (int(landmarks[end_idx-1, 0]), int(landmarks[end_idx-1, 1])),
                        (int(landmarks[start_idx, 0]), int(landmarks[start_idx, 1])),
                        color, 2)
            
            # Draw points
            for i in range(start_idx, end_idx):
                cv2.circle(vis_image, 
                          (int(landmarks[i, 0]), int(landmarks[i, 1])),
                          3, color, -1)
                
                # Add point labels for debugging
                # cv2.putText(vis_image, str(i), 
                #           (int(landmarks[i, 0]), int(landmarks[i, 1])),
                #           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return vis_image
    
    def extract_face_region(self, image: np.ndarray, landmarks: np.ndarray,
                          padding: float = 0.2) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Extract the face region from an image using landmarks.
        
        Args:
            image: Input image
            landmarks: Array of landmark points
            padding: Padding factor around the face region
            
        Returns:
            Extracted face image and face rectangle
        """
        # Get face bounds from landmarks
        min_x, min_y = np.min(landmarks, axis=0)
        max_x, max_y = np.max(landmarks, axis=0)
        
        # Add padding
        width, height = max_x - min_x, max_y - min_y
        pad_x, pad_y = int(padding * width), int(padding * height)
        
        # Calculate rectangle with padding
        x = max(0, int(min_x) - pad_x)
        y = max(0, int(min_y) - pad_y)
        w = min(image.shape[1] - x, int(max_x) - x + 2 * pad_x)
        h = min(image.shape[0] - y, int(max_y) - y + 2 * pad_y)
        
        # Extract face region
        face_rect = (x, y, w, h)
        face_image = image[y:y+h, x:x+w]
        
        return face_image, face_rect
    
    def reset(self) -> None:
        """
        Reset the tracker state.
        """
        self.prev_landmarks = None
        self.landmark_history = []
        print("Landmark tracker reset to initial state") 