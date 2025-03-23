# Avatar Animation Control System

The Avatar Animation Control System is a comprehensive framework for creating, controlling, and rendering animations for 3D avatars. It provides a high-level interface that integrates facial expressions, body animations, camera control, and environment interactions.

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Animation Script Format](#animation-script-format)
5. [Gesture Library](#gesture-library)
6. [Camera Control](#camera-control)
7. [Environment Interaction](#environment-interaction)
8. [Emotion Markup Language](#emotion-markup-language)
9. [Using the Animation Control System](#using-the-animation-control-system)
10. [Demo Script](#demo-script)

## Overview

The Animation Control System enables the creation of complex avatar animations through a script-driven approach. It supports a wide range of animation features, including facial expressions, body gestures, camera movements, and environment interactions. The system is designed to be flexible and extensible, allowing for the creation of diverse animation sequences.

## Key Features

1. **Script-to-Animation Pipeline**: Convert high-level animation scripts (in JSON format) into detailed animation sequences.

2. **Emotion Markup Language Support**: Express emotions with fine-grained control using a markup language syntax.

3. **Gesture Library with Contextual Triggering**: Predefined and custom gestures that can be triggered based on context.

4. **Environment Interaction Capabilities**: Avatars can interact with virtual objects in their environment.

5. **Virtual Camera Control**: Control camera positions, angles, and movements for optimal framing.

6. **Multi-Angle Rendering Options**: Switch between predefined camera angles or create custom camera movements.

7. **Background Integration and Compositing**: Integrate the avatar with various backgrounds (color, image, video, or 3D environment).

8. **Real-Time Preview Capabilities**: Preview animations in real-time before final rendering.

## System Architecture

The Animation Control System integrates several components:

- **AnimationControlSystem**: The main class that coordinates all animation components and processes animation scripts.

- **AvatarAnimator**: Handles the core avatar animation, including blendshapes and skeletal animation.

- **MicroExpressionSynthesizer**: Adds subtle facial movements for more natural expressions.

- **GazeController**: Controls eye movements and gaze direction.

- **HeadPoseController**: Adds natural head movements and variations.

- **EmotionController**: Manages facial expressions and emotional states.

- **GestureMannerismLearner**: Learns and reproduces person-specific gestures and mannerisms.

## Animation Script Format

The animation script is a JSON file that defines the animation sequence. It consists of sections, each representing a specific animation event:

```json
{
  "version": "1.0",
  "fps": 30,
  "avatar": "path/to/avatar.obj",
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
    // More sections...
  ]
}
```

### Section Types

1. **Speech**: Defines a speech segment with lip sync animation.
   - `start_time`: Start time in seconds
   - `end_time`: End time in seconds
   - `text`: Speech text content

2. **Emotion**: Defines an emotional expression.
   - `start_time`: Start time in seconds
   - `end_time`: End time in seconds
   - `emotion`: Emotion name (e.g., "happy", "sad", "angry")
   - `intensity`: Emotion intensity (0.0 to 1.0)
   - `transition`: Transition time in seconds

3. **Gesture**: Triggers a gesture from the gesture library.
   - `start_time`: Start time in seconds
   - `gesture`: Gesture name (e.g., "wave", "nod", "point")

4. **Camera**: Controls camera movement.
   - `start_time`: Start time in seconds
   - `action`: Camera action ("preset", "move", "zoom")
   - For preset: `preset` (preset name)
   - For move: `position` and `target` (3D coordinates)
   - For zoom: `fov` (field of view)
   - `transition`: Transition time in seconds

5. **Environment**: Controls environment interactions.
   - `start_time`: Start time in seconds
   - `action`: Environment action ("background", "interact")
   - For background: `background_type` and `value`
   - For interact: `object_id` and `interaction_type`

6. **Markup**: Uses emotion markup language for fine-grained control.
   - `start_time`: Start time in seconds
   - `markup`: Markup language string

## Gesture Library

The gesture library contains predefined and custom gestures. Each gesture is defined by:

- `type`: Gesture type (e.g., "hand", "head")
- `description`: Description of the gesture
- `duration`: Duration in seconds
- `joints`: List of joints involved
- `keyframes`: List of keyframes with time and pose data
- `context_triggers`: List of contexts that can trigger this gesture

Example gesture definition:

```json
{
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
  }
}
```

## Camera Control

The camera control system offers:

1. **Predefined Camera Presets**:
   - Front view
   - Side view
   - Three-quarter view
   - Top view
   - Custom presets

2. **Camera Movement Types**:
   - Preset: Switch to a predefined camera angle
   - Move: Move the camera to a specific position and target
   - Zoom: Change the field of view (FOV)

3. **Smooth Transitions**: All camera movements can have smooth transitions.

## Environment Interaction

The environment interaction system allows avatars to interact with virtual objects:

1. **Environment Objects**: Virtual objects that the avatar can interact with.
   - Object ID, position, rotation, scale, and model path

2. **Interaction Points**: Specific points in the environment for interaction.
   - Point ID, position, interaction type, and linked object

3. **Interaction Types**:
   - Touch
   - Grab
   - Look
   - Place
   - Custom interaction types

## Emotion Markup Language

The emotion markup language provides fine-grained control over emotions and expressions. It uses XML-like syntax:

```xml
<speech>
  <happy intensity="0.8">I'm so excited</happy> to show you these 
  <surprised intensity="0.6">amazing</surprised> features!
</speech>
```

Supported emotion tags:
- `<happy>`, `<sad>`, `<angry>`, `<surprised>`, `<disgusted>`, `<fearful>`, `<neutral>`

Attributes:
- `intensity`: Emotion intensity (0.0 to 1.0)
- `transition`: Transition time in seconds

## Using the Animation Control System

### Initialization

```python
from app.avatar_creation.animation import AnimationControlSystem

# Initialize the system
control_system = AnimationControlSystem(
    avatar_path="path/to/avatar.obj",
    output_dir="output/animations",
    use_gpu=True
)
```

### Creating an Animation Script

```python
# Create a template animation script
control_system.create_animation_script("path/to/script.json")
```

### Loading an Animation Script

```python
# Load an animation script
control_system.load_animation_script("path/to/script.json")
```

### Adding Gestures and Camera Presets

```python
# Add a custom gesture
control_system.add_gesture("gesture_name", gesture_data)

# Add a camera preset
control_system.set_camera_preset("preset_name", [x, y, z], [tx, ty, tz])
```

### Adding Environment Objects and Interaction Points

```python
# Add an environment object
control_system.add_environment_object(
    object_id="object_id",
    position=[x, y, z],
    rotation=[rx, ry, rz],
    scale=[sx, sy, sz],
    model_path="path/to/model.obj"
)

# Add an interaction point
control_system.add_interaction_point(
    point_id="point_id",
    position=[x, y, z],
    interaction_type="touch",
    linked_object="object_id"
)
```

### Stepping Through Animation

```python
# Step through the animation one frame at a time
frame_data = control_system.step_animation()
```

### Rendering Animation

```python
# Render the animation to a video file
control_system.render_animation(
    output_path="output/animation.mp4",
    start_frame=0,
    end_frame=None,  # None means all frames
    resolution=(1920, 1080)
)
```

## Demo Script

The Animation Control System includes a demo script (`demo_animation_control.py`) that demonstrates its capabilities:

```bash
python -m app.avatar_creation.animation.demo_animation_control --avatar path/to/avatar.obj
```

Command-line options:
- `--avatar`: Path to the avatar model file (required)
- `--mode`: Operation mode ("preview", "render", "script")
- `--script`: Path to animation script file
- `--output`: Path for rendered output
- `--output-dir`: Directory for output files
- `--frames`: Number of frames to preview/render (0 for all)
- `--cpu`: Force CPU usage (no GPU)

Example:
```bash
# Preview animation
python -m app.avatar_creation.animation.demo_animation_control --avatar models/avatar.obj --mode preview

# Render animation
python -m app.avatar_creation.animation.demo_animation_control --avatar models/avatar.obj --mode render --output animation.mp4

# Use a custom script
python -m app.avatar_creation.animation.demo_animation_control --avatar models/avatar.obj --script my_script.json
``` 