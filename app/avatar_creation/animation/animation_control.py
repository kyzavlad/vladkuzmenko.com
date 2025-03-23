import os
import json
import numpy as np
import time
import math
from typing import Dict, List, Tuple, Optional, Union, Any

from app.avatar_creation.animation.avatar_animator import AvatarAnimator
from app.avatar_creation.animation.micro_expression import MicroExpressionSynthesizer
from app.avatar_creation.animation.gaze_controller import GazeController
from app.avatar_creation.animation.head_pose import HeadPoseController
from app.avatar_creation.animation.emotion_controller import EmotionController
from app.avatar_creation.animation.gesture_model import GestureMannerismLearner, GestureModelConfig
from app.avatar_creation.animation.markup_parser import EmotionMarkupParser
from app.avatar_creation.animation.camera_path import CameraPath, CameraKeyframe, CameraPathLibrary
from app.avatar_creation.face_modeling.utils import ensure_directory

class AnimationControlSystem:
    """
    High-level control system for avatar animation.
    
    Features:
    - Script-to-animation pipeline
    - Emotion markup language support
    - Gesture library with contextual triggering
    - Environment interaction capabilities
    - Virtual camera control
    - Multi-angle rendering
    - Background integration
    - Real-time preview
    """
    
    def __init__(self,
                avatar_path: str,
                output_dir: str = "output/animations",
                use_gpu: bool = True):
        """
        Initialize the animation control system.
        
        Args:
            avatar_path: Path to the avatar model
            output_dir: Directory for output files
            use_gpu: Whether to use GPU acceleration
        """
        self.avatar_path = avatar_path
        self.output_dir = output_dir
        self.use_gpu = use_gpu
        
        # Ensure output directory exists
        ensure_directory(output_dir)
        
        # Initialize animation components
        self._initialize_components()
        
        # Animation script and sequence data
        self.current_script = None
        self.animation_sequence = []
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        
        # Camera and rendering settings
        self.camera_settings = {
            "position": np.array([0.0, 0.0, 1.0]),
            "target": np.array([0.0, 0.0, 0.0]),
            "up": np.array([0.0, 1.0, 0.0]),
            "fov": 45.0,
            "near": 0.1,
            "far": 1000.0
        }
        
        # Initialize camera path library
        self.camera_path_library = CameraPathLibrary()
        self.camera_path_library.create_common_paths()
        self.camera_paths = {name: path for name, path in self.camera_path_library.paths.items()}
        
        # Load camera presets from the library
        self.camera_presets = self.camera_path_library.presets.copy()
        
        # Background and compositing settings
        self.background_settings = {
            "type": "color",  # color, image, video, or environment
            "color": np.array([0.0, 0.0, 0.0, 1.0]),
            "path": None,
            "blur": 0.0,
            "brightness": 1.0
        }
        
        # Environment interaction settings
        self.environment_objects = {}
        self.interaction_points = {}
        
        # Gesture library
        self.gesture_library = {}
        self._load_default_gestures()
        
        print(f"Animation Control System initialized")
        print(f"  - Avatar: {os.path.basename(avatar_path)}")
        print(f"  - Output directory: {output_dir}")
    
    def _initialize_components(self) -> None:
        """Initialize all animation components."""
        # Avatar animator (core animation engine)
        self.animator = AvatarAnimator(
            model_path=self.avatar_path,
            use_gpu=self.use_gpu
        )
        
        # Micro-expression synthesizer (for natural facial movements)
        self.micro_expression = MicroExpressionSynthesizer()
        
        # Gaze controller (for natural eye movements)
        self.gaze_controller = GazeController()
        
        # Head pose controller (for natural head movements)
        self.head_pose = HeadPoseController()
        
        # Emotion controller (for facial expressions)
        self.emotion_controller = EmotionController()
        
        # Gesture model (for body language)
        gesture_config = GestureModelConfig(use_gpu=self.use_gpu)
        self.gesture_model = GestureMannerismLearner(config=gesture_config)
    
    def _load_default_gestures(self) -> None:
        """Load default gesture library."""
        # Basic gestures
        self.gesture_library = {
            "wave": {
                "type": "hand",
                "description": "Wave hand greeting",
                "duration": 2.0,
                "joints": ["shoulder_r", "elbow_r", "wrist_r"],
                "keyframes": [
                    {"time": 0.0, "pose": {"shoulder_r": [0, 0, 0], "elbow_r": [0, 0, 0], "wrist_r": [0, 0, 0]}},
                    {"time": 0.5, "pose": {"shoulder_r": [0, 30, 0], "elbow_r": [0, 0, 80], "wrist_r": [0, 0, 20]}},
                    {"time": 1.0, "pose": {"shoulder_r": [0, 30, 0], "elbow_r": [0, 0, 80], "wrist_r": [0, 0, -20]}},
                    {"time": 1.5, "pose": {"shoulder_r": [0, 30, 0], "elbow_r": [0, 0, 80], "wrist_r": [0, 0, 20]}},
                    {"time": 2.0, "pose": {"shoulder_r": [0, 0, 0], "elbow_r": [0, 0, 0], "wrist_r": [0, 0, 0]}}
                ],
                "context_triggers": ["greeting", "goodbye"]
            },
            "nod": {
                "type": "head",
                "description": "Head nodding (yes)",
                "duration": 1.5,
                "joints": ["head"],
                "keyframes": [
                    {"time": 0.0, "pose": {"head": [0, 0, 0]}},
                    {"time": 0.3, "pose": {"head": [20, 0, 0]}},
                    {"time": 0.6, "pose": {"head": [0, 0, 0]}},
                    {"time": 0.9, "pose": {"head": [10, 0, 0]}},
                    {"time": 1.2, "pose": {"head": [0, 0, 0]}}
                ],
                "context_triggers": ["agreement", "acknowledgment", "approval"]
            },
            "head_shake": {
                "type": "head",
                "description": "Head shaking (no)",
                "duration": 1.5,
                "joints": ["head"],
                "keyframes": [
                    {"time": 0.0, "pose": {"head": [0, 0, 0]}},
                    {"time": 0.25, "pose": {"head": [0, -15, 0]}},
                    {"time": 0.5, "pose": {"head": [0, 15, 0]}},
                    {"time": 0.75, "pose": {"head": [0, -15, 0]}},
                    {"time": 1.0, "pose": {"head": [0, 15, 0]}},
                    {"time": 1.5, "pose": {"head": [0, 0, 0]}}
                ],
                "context_triggers": ["disagreement", "rejection", "denial"]
            },
            "point": {
                "type": "hand",
                "description": "Pointing gesture",
                "duration": 1.0,
                "joints": ["shoulder_r", "elbow_r", "wrist_r", "index_finger_r"],
                "keyframes": [
                    {"time": 0.0, "pose": {"shoulder_r": [0, 0, 0], "elbow_r": [0, 0, 0], "wrist_r": [0, 0, 0], "index_finger_r": [0, 0, 0]}},
                    {"time": 0.5, "pose": {"shoulder_r": [0, 45, 0], "elbow_r": [0, 0, 40], "wrist_r": [0, 0, 0], "index_finger_r": [0, 0, 20]}},
                    {"time": 1.0, "pose": {"shoulder_r": [0, 0, 0], "elbow_r": [0, 0, 0], "wrist_r": [0, 0, 0], "index_finger_r": [0, 0, 0]}}
                ],
                "context_triggers": ["indicating", "direction", "reference"]
            }
        }
    
    def load_animation_script(self, script_path: str) -> bool:
        """
        Load an animation script from a file.
        
        Args:
            script_path: Path to the animation script file (JSON format)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(script_path, 'r') as f:
                script_data = json.load(f)
            
            self.current_script = script_data
            
            # Parse the script into an animation sequence
            self._parse_script_to_animation()
            
            print(f"Loaded animation script: {script_path}")
            print(f"  - Total frames: {self.total_frames}")
            print(f"  - Duration: {self.total_frames / self.fps:.2f} seconds")
            
            return True
        except Exception as e:
            print(f"Error loading animation script: {e}")
            return False
    
    def _parse_script_to_animation(self) -> None:
        """Parse the current script into an animation sequence."""
        if not self.current_script:
            return
        
        # Reset animation sequence
        self.animation_sequence = []
        self.current_frame = 0
        
        # Get FPS from script or use default
        self.fps = self.current_script.get("fps", 30)
        
        # Process script sections
        for section in self.current_script.get("sections", []):
            section_type = section.get("type", "")
            
            if section_type == "speech":
                self._parse_speech_section(section)
            elif section_type == "emotion":
                self._parse_emotion_section(section)
            elif section_type == "gesture":
                self._parse_gesture_section(section)
            elif section_type == "camera":
                self._parse_camera_section(section)
            elif section_type == "environment":
                self._parse_environment_section(section)
            elif section_type == "markup":
                self._parse_markup_section(section)
        
        # Sort the animation sequence by frame
        self.animation_sequence.sort(key=lambda x: x.get("frame", 0))
        
        # Set total frames
        if self.animation_sequence:
            self.total_frames = max(item.get("frame", 0) for item in self.animation_sequence) + 1
        else:
            self.total_frames = 0
    
    def _parse_speech_section(self, section: Dict) -> None:
        """Parse a speech section into animation keyframes."""
        start_time = section.get("start_time", 0)
        end_time = section.get("end_time", start_time + 5)
        text = section.get("text", "")
        
        # Convert time to frames
        start_frame = int(start_time * self.fps)
        end_frame = int(end_time * self.fps)
        
        # Add lip sync keyframes (placeholder - would use actual lip sync in production)
        for frame in range(start_frame, end_frame):
            # Simple oscillating mouth movement as placeholder
            openness = 0.2 + 0.3 * abs(math.sin((frame - start_frame) * 0.2))
            
            self.animation_sequence.append({
                "frame": frame,
                "type": "blendshape",
                "name": "mouth_open",
                "value": openness
            })
    
    def _parse_emotion_section(self, section: Dict) -> None:
        """Parse an emotion section into animation keyframes."""
        start_time = section.get("start_time", 0)
        end_time = section.get("end_time", start_time + 2)
        emotion = section.get("emotion", "neutral")
        intensity = section.get("intensity", 1.0)
        transition = section.get("transition", 0.5)
        
        # Convert time to frames
        start_frame = int(start_time * self.fps)
        end_frame = int(end_time * self.fps)
        transition_frames = int(transition * self.fps)
        
        # Add emotion keyframes
        self.animation_sequence.append({
            "frame": start_frame,
            "type": "emotion",
            "name": emotion,
            "value": intensity,
            "transition_frames": transition_frames
        })
        
        # Add reset keyframe if specified
        if section.get("reset", False):
            self.animation_sequence.append({
                "frame": end_frame,
                "type": "emotion",
                "name": "neutral",
                "value": 1.0,
                "transition_frames": transition_frames
            })
    
    def _parse_gesture_section(self, section: Dict) -> None:
        """Parse a gesture section into animation keyframes."""
        start_time = section.get("start_time", 0)
        gesture_name = section.get("gesture", "")
        
        # Look up the gesture in our library
        if gesture_name in self.gesture_library:
            gesture = self.gesture_library[gesture_name]
            duration = gesture.get("duration", 1.0)
            
            # Convert time to frames
            start_frame = int(start_time * self.fps)
            
            # Add gesture keyframes
            for keyframe in gesture.get("keyframes", []):
                keyframe_time = keyframe.get("time", 0)
                keyframe_frame = start_frame + int(keyframe_time * self.fps)
                
                # Process each joint in the keyframe
                for joint, rotation in keyframe.get("pose", {}).items():
                    self.animation_sequence.append({
                        "frame": keyframe_frame,
                        "type": "bone",
                        "name": joint,
                        "value": rotation
                    })
    
    def _parse_camera_section(self, section: Dict) -> None:
        """Parse a camera section into animation keyframes."""
        start_time = section.get("start_time", 0)
        camera_action = section.get("action", "")
        
        # Convert time to frames
        start_frame = int(start_time * self.fps)
        
        if camera_action == "preset":
            # Use a predefined camera preset
            preset_name = section.get("preset", "front")
            if preset_name in self.camera_presets:
                preset = self.camera_presets[preset_name]
                
                self.animation_sequence.append({
                    "frame": start_frame,
                    "type": "camera",
                    "action": "move",
                    "position": preset["position"].tolist(),
                    "target": preset["target"].tolist(),
                    "transition_frames": int(section.get("transition", 1.0) * self.fps)
                })
        
        elif camera_action == "move":
            # Custom camera movement
            self.animation_sequence.append({
                "frame": start_frame,
                "type": "camera",
                "action": "move",
                "position": section.get("position", [0, 0, 1]),
                "target": section.get("target", [0, 0, 0]),
                "transition_frames": int(section.get("transition", 1.0) * self.fps)
            })
        
        elif camera_action == "zoom":
            # Camera zoom
            self.animation_sequence.append({
                "frame": start_frame,
                "type": "camera",
                "action": "zoom",
                "fov": section.get("fov", 45.0),
                "transition_frames": int(section.get("transition", 1.0) * self.fps)
            })
            
        elif camera_action == "path":
            # Use a camera path
            path_name = section.get("path_name", "")
            if path_name in self.camera_paths:
                path = self.camera_paths[path_name]
                duration = section.get("duration", path.duration)
                
                # Add camera path to animation sequence
                self.animation_sequence.append({
                    "frame": start_frame,
                    "type": "camera",
                    "action": "path",
                    "path_name": path_name,
                    "duration": duration,
                    "loop": section.get("loop", path.loop)
                })
    
    def _parse_environment_section(self, section: Dict) -> None:
        """Parse an environment section into animation keyframes."""
        start_time = section.get("start_time", 0)
        action = section.get("action", "")
        
        # Convert time to frames
        start_frame = int(start_time * self.fps)
        
        if action == "background":
            # Change background
            self.animation_sequence.append({
                "frame": start_frame,
                "type": "environment",
                "action": "background",
                "background_type": section.get("background_type", "color"),
                "value": section.get("value", [0, 0, 0, 1])
            })
        
        elif action == "interact":
            # Interact with environment object
            self.animation_sequence.append({
                "frame": start_frame,
                "type": "environment",
                "action": "interact",
                "object_id": section.get("object_id", ""),
                "interaction_type": section.get("interaction_type", "touch")
            })
    
    def _parse_markup_section(self, section: Dict) -> None:
        """Parse an emotion markup language section."""
        start_time = section.get("start_time", 0)
        markup = section.get("markup", "")
        
        # Convert time to frames
        start_frame = int(start_time * self.fps)
        
        # Use the EmotionMarkupParser to parse the markup
        parser = EmotionMarkupParser()
        animation_events = parser.convert_to_animation_events(
            markup, 
            start_time=start_time,
            fps=self.fps
        )
        
        # Add all parsed emotion events to the animation sequence
        for event in animation_events:
            event_start_frame = int(event["start_time"] * self.fps)
            event_end_frame = int(event["end_time"] * self.fps)
            transition_frames = int(event["transition"] * self.fps)
            
            self.animation_sequence.append({
                "frame": event_start_frame,
                "type": "emotion",
                "name": event["emotion"],
                "value": event["intensity"],
                "transition_frames": transition_frames,
                "end_frame": event_end_frame
            })
            
        # Also extract the plain text for speech synthesis
        plain_text = parser.extract_text(markup)
        if plain_text.strip():
            # Add speech section for the plain text
            self.animation_sequence.append({
                "frame": start_frame,
                "type": "speech",
                "text": plain_text,
                "end_frame": int((start_time + len(plain_text) * 0.07) * self.fps)  # Approximate timing
            })
    
    def create_animation_script(self, output_path: str) -> None:
        """
        Create a template animation script file.
        
        Args:
            output_path: Path to save the template script
        """
        template_script = {
            "version": "1.0",
            "fps": 30,
            "avatar": self.avatar_path,
            "sections": [
                {
                    "type": "speech",
                    "start_time": 0.0,
                    "end_time": 4.0,
                    "text": "Hello! This is an example animation script."
                },
                {
                    "type": "emotion",
                    "start_time": 0.5,
                    "end_time": 2.5,
                    "emotion": "happy",
                    "intensity": 0.8,
                    "transition": 0.5
                },
                {
                    "type": "gesture",
                    "start_time": 1.0,
                    "gesture": "wave"
                },
                {
                    "type": "camera",
                    "start_time": 0.0,
                    "action": "preset",
                    "preset": "front"
                },
                {
                    "type": "camera",
                    "start_time": 2.0,
                    "action": "preset",
                    "preset": "three-quarter",
                    "transition": 1.0
                },
                {
                    "type": "environment",
                    "start_time": 0.0,
                    "action": "background",
                    "background_type": "color",
                    "value": [0.2, 0.2, 0.8, 1.0]
                },
                {
                    "type": "markup",
                    "start_time": 3.0,
                    "markup": "<speech><happy>This is emotion markup language.</happy></speech>"
                }
            ]
        }
        
        # Save the template script
        with open(output_path, 'w') as f:
            json.dump(template_script, f, indent=2)
            
        print(f"Created template animation script: {output_path}")
    
    def step_animation(self) -> Dict:
        """
        Step forward one frame in the animation.
        
        Returns:
            Frame data dictionary for rendering
        """
        # Initialize frame data
        frame_data = {
            "frame": self.current_frame,
            "time": self.current_frame / self.fps,
            "blendshapes": {},
            "bones": {},
            "camera": self.camera_settings.copy(),
            "background": self.background_settings.copy()
        }
        
        # Process all animation events for this frame
        for event in [e for e in self.animation_sequence if e.get("frame") == self.current_frame]:
            event_type = event.get("type", "")
            
            if event_type == "blendshape":
                frame_data["blendshapes"][event["name"]] = event["value"]
            
            elif event_type == "bone":
                frame_data["bones"][event["name"]] = event["value"]
            
            elif event_type == "emotion":
                # Set emotion in the emotion controller
                self.emotion_controller.set_emotion(
                    event["name"], 
                    event["value"]
                )
            
            elif event_type == "camera":
                if event["action"] == "move":
                    frame_data["camera"]["position"] = event["position"]
                    frame_data["camera"]["target"] = event["target"]
                elif event["action"] == "zoom":
                    frame_data["camera"]["fov"] = event["fov"]
                elif event["action"] == "path":
                    # Handle camera path event
                    self._process_camera_path_event(event, frame_data)
            
            elif event_type == "environment":
                if event["action"] == "background":
                    frame_data["background"]["type"] = event["background_type"]
                    frame_data["background"]["color"] = event["value"]
        
        # Add micro-expressions and natural movements
        micro_expressions = self.micro_expression.update(1.0 / self.fps)
        for expr, value in micro_expressions.items():
            if expr not in frame_data["blendshapes"]:
                frame_data["blendshapes"][expr] = value
        
        # Add eye gaze
        gaze_data = self.gaze_controller.update(1.0 / self.fps)
        for expr, value in gaze_data.get("blendshapes", {}).items():
            if expr not in frame_data["blendshapes"]:
                frame_data["blendshapes"][expr] = value
        
        # Add head pose variations
        head_data = self.head_pose.update(1.0 / self.fps)
        for bone, rotation in head_data.get("bones", {}).items():
            if bone not in frame_data["bones"]:
                frame_data["bones"][bone] = rotation
        
        # Update emotion controller
        emotion_data = self.emotion_controller.update(1.0 / self.fps)
        emotion_blendshapes = self._emotions_to_blendshapes(emotion_data)
        for expr, value in emotion_blendshapes.items():
            if expr not in frame_data["blendshapes"]:
                frame_data["blendshapes"][expr] = value
        
        # Apply frame data to the avatar
        for name, value in frame_data["blendshapes"].items():
            self.animator.set_blendshape_weight(name, value)
            
        for name, rotation in frame_data["bones"].items():
            self.animator.set_bone_rotation(name, np.array(rotation))
        
        # Update the animator
        avatar_data = self.animator.update()
        frame_data["avatar"] = avatar_data
        
        # Increment frame counter
        self.current_frame += 1
        if self.current_frame >= self.total_frames:
            self.current_frame = 0  # Loop animation
        
        return frame_data
    
    def _process_camera_path_event(self, event: Dict, frame_data: Dict) -> None:
        """
        Process a camera path event and update frame data.
        
        Args:
            event: Camera path event data
            frame_data: Frame data to update
        """
        path_name = event.get("path_name", "")
        if path_name not in self.camera_paths:
            return
            
        path = self.camera_paths[path_name]
        duration = event.get("duration", path.duration)
        
        # Calculate time within the path
        event_start_frame = event.get("frame", 0)
        elapsed_frames = self.current_frame - event_start_frame
        elapsed_time = elapsed_frames / self.fps
        
        # Handle looping if needed
        if event.get("loop", path.loop) and elapsed_time > duration:
            elapsed_time = elapsed_time % duration
        
        # Get camera properties at the current time in the path
        if elapsed_time <= duration:
            # Scale the elapsed time to fit the path duration
            path_time = elapsed_time * (path.duration / duration)
            camera_props = path.get_camera_at_time(path_time)
            
            # Update frame data with camera properties
            frame_data["camera"]["position"] = camera_props["position"]
            frame_data["camera"]["target"] = camera_props["target"]
            frame_data["camera"]["up"] = camera_props["up"]
            frame_data["camera"]["fov"] = camera_props["fov"]
    
    def _emotions_to_blendshapes(self, emotion_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Convert emotion weights to blendshape weights.
        
        Args:
            emotion_weights: Dictionary of emotion weights
            
        Returns:
            Dictionary of blendshape weights
        """
        blendshapes = {}
        
        # Basic emotion to blendshape mapping
        emotion_map = {
            'happy': ['smile', 'cheek_raise', 'eye_squint'],
            'sad': ['brow_lower', 'mouth_frown', 'eye_close'],
            'angry': ['brow_lower', 'eye_squint', 'mouth_narrow'],
            'surprised': ['brow_raise', 'eye_wide', 'mouth_open'],
            'disgusted': ['nose_wrinkle', 'upper_lip_raise', 'brow_lower'],
            'fearful': ['eye_wide', 'brow_raise', 'mouth_open_slight'],
            'neutral': ['face_neutral']
        }
        
        # Apply emotion to blendshape mapping
        for emotion, weight in emotion_weights.items():
            if emotion in emotion_map:
                for blendshape in emotion_map[emotion]:
                    blendshapes[blendshape] = blendshapes.get(blendshape, 0.0) + weight
        
        # Normalize blendshape values
        for name in blendshapes:
            blendshapes[name] = min(1.0, blendshapes[name])
        
        return blendshapes
    
    def add_gesture(self, gesture_name: str, gesture_data: Dict) -> None:
        """
        Add a new gesture to the library.
        
        Args:
            gesture_name: Name of the gesture
            gesture_data: Gesture data dictionary
        """
        self.gesture_library[gesture_name] = gesture_data
        print(f"Added gesture '{gesture_name}' to library")
    
    def set_camera_preset(self, preset_name: str, position: List[float], target: List[float]) -> None:
        """
        Add or update a camera preset.
        
        Args:
            preset_name: Name of the preset
            position: Camera position [x, y, z]
            target: Camera target [x, y, z]
        """
        self.camera_presets[preset_name] = {
            "position": np.array(position),
            "target": np.array(target)
        }
        print(f"Set camera preset '{preset_name}'")
    
    def add_environment_object(self, 
                              object_id: str, 
                              position: List[float], 
                              rotation: List[float],
                              scale: List[float],
                              model_path: str) -> None:
        """
        Add an environment object for interaction.
        
        Args:
            object_id: Unique ID for the object
            position: Object position [x, y, z]
            rotation: Object rotation [x, y, z]
            scale: Object scale [x, y, z]
            model_path: Path to the 3D model
        """
        self.environment_objects[object_id] = {
            "position": np.array(position),
            "rotation": np.array(rotation),
            "scale": np.array(scale),
            "model_path": model_path
        }
        print(f"Added environment object '{object_id}'")
    
    def add_interaction_point(self, 
                             point_id: str, 
                             position: List[float],
                             interaction_type: str,
                             linked_object: Optional[str] = None) -> None:
        """
        Add an interaction point in the environment.
        
        Args:
            point_id: Unique ID for the interaction point
            position: Point position [x, y, z]
            interaction_type: Type of interaction (e.g., "touch", "grab", "look")
            linked_object: Optional ID of a linked environment object
        """
        self.interaction_points[point_id] = {
            "position": np.array(position),
            "interaction_type": interaction_type,
            "linked_object": linked_object
        }
        print(f"Added interaction point '{point_id}'")
    
    def render_animation(self, 
                        output_path: str, 
                        start_frame: int = 0,
                        end_frame: Optional[int] = None,
                        resolution: Tuple[int, int] = (1920, 1080)) -> None:
        """
        Render the animation to a video file.
        
        Args:
            output_path: Path to save the rendered video
            start_frame: First frame to render
            end_frame: Last frame to render (None to render all frames)
            resolution: Output resolution (width, height)
        """
        if end_frame is None:
            end_frame = self.total_frames
        
        print(f"Rendering animation to {output_path}")
        print(f"  - Frames: {start_frame} to {end_frame}")
        print(f"  - Resolution: {resolution[0]}x{resolution[1]}")
        
        # This would integrate with a rendering engine in a full implementation
        # For now, we'll just simulate the rendering process
        
        self.current_frame = start_frame
        
        # Placeholder for render loop
        for frame in range(start_frame, end_frame):
            frame_data = self.step_animation()
            # In a real implementation, this would render the frame
            
            # Simulate rendering progress
            if frame % 10 == 0:
                progress = (frame - start_frame) / (end_frame - start_frame) * 100
                print(f"Rendering: {progress:.1f}% complete")
        
        print(f"Animation rendering complete: {output_path}")
    
    def reset(self) -> None:
        """Reset the animation control system to initial state."""
        self.current_script = None
        self.animation_sequence = []
        self.current_frame = 0
        self.total_frames = 0
        
        # Reset all animation components
        self.animator.reset()
        self.micro_expression.reset()
        self.gaze_controller.reset()
        self.head_pose.reset()
        self.emotion_controller.reset()
        self.gesture_model.reset()
        
        print("Animation control system reset to initial state")
    
    def add_camera_path(self, path: CameraPath) -> None:
        """
        Add a camera path to the system.
        
        Args:
            path: Camera path to add
        """
        self.camera_paths[path.name] = path
        print(f"Added camera path '{path.name}'")
    
    def remove_camera_path(self, name: str) -> bool:
        """
        Remove a camera path from the system.
        
        Args:
            name: Name of the camera path to remove
            
        Returns:
            True if removed, False if not found
        """
        if name in self.camera_paths:
            del self.camera_paths[name]
            print(f"Removed camera path '{name}'")
            return True
        return False
    
    def create_camera_path(self, 
                         name: str, 
                         path_type: str = "orbit", 
                         target: List[float] = None,
                         **kwargs) -> CameraPath:
        """
        Create a new camera path.
        
        Args:
            name: Name for the camera path
            path_type: Type of path ('orbit', 'flyby', 'custom')
            target: Target to look at, defaults to origin
            **kwargs: Additional arguments for the path
            
        Returns:
            The created camera path
        """
        if target is None:
            target = [0.0, 0.0, 0.0]
            
        path = CameraPath(name)
        
        if path_type == "orbit":
            radius = kwargs.get("radius", 1.0)
            height = kwargs.get("height", 0.2)
            duration = kwargs.get("duration", 10.0)
            num_keyframes = kwargs.get("num_keyframes", 24)
            
            path.create_orbit_path(
                center=target,
                radius=radius,
                height=height,
                duration=duration,
                num_keyframes=num_keyframes
            )
            
        elif path_type == "flyby":
            start_position = kwargs.get("start_position", [-1.0, 0.3, 1.0])
            end_position = kwargs.get("end_position", [1.0, 0.3, 1.0])
            duration = kwargs.get("duration", 5.0)
            curve_height = kwargs.get("curve_height", 0.2)
            
            path.create_flyby_path(
                start_position=start_position,
                end_position=end_position,
                target=target,
                duration=duration,
                curve_height=curve_height
            )
        
        # Add the path to the system
        self.add_camera_path(path)
        
        return path
    
    def import_camera_library(self, file_path: str) -> bool:
        """
        Import camera paths and presets from a library file.
        
        Args:
            file_path: Path to the library file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            library = CameraPathLibrary.load_library(file_path)
            
            # Add paths from the library
            for name, path in library.paths.items():
                self.camera_paths[name] = path
            
            # Add presets from the library
            for name, preset in library.presets.items():
                self.camera_presets[name] = preset
            
            print(f"Imported camera library from {file_path}")
            print(f"  - Paths: {len(library.paths)}")
            print(f"  - Presets: {len(library.presets)}")
            
            return True
        except Exception as e:
            print(f"Error importing camera library: {e}")
            return False
    
    def export_camera_library(self, file_path: str) -> bool:
        """
        Export camera paths and presets to a library file.
        
        Args:
            file_path: Path to save the library file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            library = CameraPathLibrary()
            
            # Add paths to the library
            for name, path in self.camera_paths.items():
                library.add_path(path)
            
            # Add presets to the library
            library.presets = self.camera_presets.copy()
            
            # Save the library
            library.save_library(file_path)
            
            print(f"Exported camera library to {file_path}")
            print(f"  - Paths: {len(self.camera_paths)}")
            print(f"  - Presets: {len(self.camera_presets)}")
            
            return True
        except Exception as e:
            print(f"Error exporting camera library: {e}")
            return False 