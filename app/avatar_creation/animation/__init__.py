"""
Avatar Animation Module

This module provides classes and functions for real-time animation of 3D avatars,
including facial animation, body motion, gesture synthesis, and emotion expression.
It supports various animation techniques including:

1. First Order Motion Model for video-driven animation with temporal consistency
2. Facial expression and micro-expression synthesis
3. Gaze direction control with natural eye movements
4. Head pose variation with natural movement patterns
5. Emotion control with natural transitions
6. Person-specific gesture and mannerism learning
7. Script-to-animation pipeline with emotion markup language
8. Environment interaction and virtual camera control
9. Camera path editing for cinematic animation sequences
10. Gesture building tools for creating custom avatar gestures

The module is designed to provide both high-level animation control and
fine-grained control over specific aspects of avatar animation.
"""

# Import classes from submodules
from app.avatar_creation.animation.avatar_animator import AvatarAnimator
from app.avatar_creation.animation.facial_landmark import FacialLandmarkTracker
from app.avatar_creation.animation.micro_expression import MicroExpressionSynthesizer
from app.avatar_creation.animation.gaze_controller import GazeController
from app.avatar_creation.animation.head_pose import HeadPoseController
from app.avatar_creation.animation.emotion_controller import EmotionController
from app.avatar_creation.animation.motion_model import FirstOrderMotionModel, MotionModelConfig
from app.avatar_creation.animation.gesture_model import GestureMannerismLearner, GestureModelConfig
from app.avatar_creation.animation.animation_control import AnimationControlSystem
from app.avatar_creation.animation.markup_parser import EmotionMarkupParser
from app.avatar_creation.animation.gesture_builder import GestureBuilder
from app.avatar_creation.animation.camera_path import CameraPath, CameraKeyframe, CameraPathLibrary

# Define the public API
__all__ = [
    'AvatarAnimator',
    'FacialLandmarkTracker',
    'MicroExpressionSynthesizer',
    'GazeController',
    'HeadPoseController',
    'EmotionController',
    'FirstOrderMotionModel',
    'MotionModelConfig',
    'GestureMannerismLearner',
    'GestureModelConfig',
    'AnimationControlSystem',
    'EmotionMarkupParser',
    'GestureBuilder',
    'CameraPath',
    'CameraKeyframe',
    'CameraPathLibrary'
] 