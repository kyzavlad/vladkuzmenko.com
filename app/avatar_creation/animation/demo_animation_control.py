#!/usr/bin/env python3
"""
Avatar Animation Control System Demo

This script demonstrates the capabilities of the Avatar Animation Control System,
including:
- Script-to-animation pipeline
- Emotion markup language support
- Gesture library with contextual triggering
- Environment interaction capabilities
- Virtual camera control for optimal framing
- Multi-angle rendering options
- Background integration and compositing
- Real-time preview capabilities

Usage: python -m app.avatar_creation.animation.demo_animation_control
"""

import os
import sys
import time
import argparse
import json
from app.avatar_creation.animation import AnimationControlSystem
from app.avatar_creation.animation.markup_parser import EmotionMarkupParser
from app.avatar_creation.animation.gesture_builder import GestureBuilder
from app.avatar_creation.animation.camera_path import CameraPath, CameraKeyframe
from app.avatar_creation.face_modeling.utils import ensure_directory

def demo_animation_control(args):
    """
    Demonstrate the animation control system with the given arguments.
    
    Args:
        args: Command-line arguments
    """
    print("\n=== Avatar Animation Control System Demo ===\n")
    
    # Initialize the animation control system
    control_system = AnimationControlSystem(
        avatar_path=args.avatar_path,
        output_dir=args.output_dir,
        use_gpu=not args.cpu
    )
    
    # Run appropriate demo based on mode
    if args.mode == "emotion_markup":
        demo_emotion_markup(control_system, args)
    elif args.mode == "gesture_builder":
        demo_gesture_builder(control_system, args)
    elif args.mode == "camera_path":
        demo_camera_path(control_system, args)
    else:
        # Default mode
        run_standard_demo(control_system, args)

def run_standard_demo(control_system, args):
    """Run the standard animation demo with script loading and playback."""
    # Create a sample script if not provided
    script_path = args.script
    if script_path is None:
        script_path = os.path.join(args.output_dir, "generated_script.json")
        control_system.create_animation_script(script_path)
        print(f"Created sample animation script: {script_path}")
    
    # Load the animation script
    if not control_system.load_animation_script(script_path):
        print("Failed to load animation script. Exiting.")
        return
    
    # Add environment objects for interaction
    demo_add_environment_objects(control_system)
    
    # Add custom gesture
    demo_add_custom_gesture(control_system)
    
    # Add custom camera preset
    control_system.set_camera_preset(
        "dramatic_low", 
        [0.5, -0.3, 0.8], 
        [0.0, 0.2, 0.0]
    )
    
    # Create custom camera paths
    create_demo_camera_paths(control_system)
    
    # Perform the requested operation
    if args.preview:
        demo_preview_animation(control_system, args)
    elif args.render:
        demo_render_animation(control_system, args)
    else:
        print(f"Script loaded successfully: {script_path}")
        print(f"Total frames: {control_system.total_frames}")
        print(f"Duration: {control_system.total_frames / control_system.fps:.2f} seconds")
        
        # Show available camera paths
        print("\nAvailable camera paths:")
        for path_name in control_system.camera_paths:
            path = control_system.camera_paths[path_name]
            print(f"  - {path_name} (duration: {path.duration:.1f}s, {len(path.keyframes)} keyframes)")

def demo_emotion_markup(control_system, args):
    """Demonstrate emotion markup language parsing."""
    print("\n--- Emotion Markup Language Demo ---\n")
    
    # Create an emotion markup parser
    parser = EmotionMarkupParser()
    
    # Sample markup text
    markup_text = """
    <happy intensity="0.8" transition="0.5">I'm so excited</happy> to show you these 
    <surprised intensity="0.6">amazing</surprised> features! When things go wrong, I might look
    <sad intensity="0.7" transition="0.3">a bit disappointed</sad>, but I can also express
    <angry intensity="0.5">frustration</angry> or even <fearful intensity="0.6">fear</fearful>.
    """
    
    # Parse the markup
    plain_text, emotion_events = parser.parse(markup_text)
    
    # Display the parsed results
    print("Original markup:")
    print(markup_text)
    print("\nPlain text:")
    print(plain_text)
    print("\nEmotion events:")
    for event in emotion_events:
        text_range = event['text_range']
        text_snippet = plain_text[text_range[0]:text_range[1]]
        print(f"  - {event['emotion']} (intensity: {event['intensity']:.1f}, transition: {event['transition']:.1f}s)")
        print(f"    Text: \"{text_snippet}\"")
    
    # Convert to animation events
    animation_events = parser.convert_to_animation_events(markup_text, start_time=1.0)
    
    print("\nAnimation events:")
    for event in animation_events:
        print(f"  - {event['emotion']} from {event['start_time']:.2f}s to {event['end_time']:.2f}s (intensity: {event['intensity']:.1f})")
    
    # Create a simple script using the markup
    if args.preview or args.render:
        script_path = os.path.join(args.output_dir, "markup_demo.json")
        
        # Create a script with the markup
        script = {
            "version": "1.0",
            "fps": 30,
            "avatar": args.avatar_path,
            "sections": [
                {
                    "type": "markup",
                    "start_time": 1.0,
                    "markup": markup_text
                },
                {
                    "type": "camera",
                    "start_time": 0.0,
                    "action": "preset",
                    "preset": "front"
                }
            ]
        }
        
        # Save the script
        with open(script_path, 'w') as f:
            json.dump(script, f, indent=2)
        
        print(f"\nCreated markup demo script: {script_path}")
        
        # Load and preview/render the script
        control_system.load_animation_script(script_path)
        
        if args.preview:
            demo_preview_animation(control_system, args)
        elif args.render:
            demo_render_animation(control_system, args)

def demo_gesture_builder(control_system, args):
    """Demonstrate gesture builder functionality."""
    print("\n--- Gesture Builder Demo ---\n")
    
    # Create a gesture builder
    builder = GestureBuilder()
    
    # Show available templates
    templates = builder.get_available_templates()
    print("Available gesture templates:")
    for template in templates:
        print(f"  - {template}")
    
    # Create a few gestures
    print("\nCreating custom gesture:")
    
    # 1. Create a thinking gesture
    builder.create_new_gesture(
        name="thoughtful_pose",
        gesture_type="upper_body",
        description="Thoughtful pose with hand on chin and occasional head tilt",
        duration=4.0
    )
    
    # Add joints
    for joint in ["shoulder_r", "elbow_r", "wrist_r", "head"]:
        builder.add_joint(joint)
    
    # Add keyframes
    builder.add_keyframe(1.0)
    builder.add_keyframe(2.0)
    builder.add_keyframe(3.0)
    
    # Set joint rotations
    builder.set_joint_rotation(1.0, "shoulder_r", [0, 15, 0])
    builder.set_joint_rotation(1.0, "elbow_r", [0, 0, 90])
    builder.set_joint_rotation(1.0, "wrist_r", [0, 0, 30])
    builder.set_joint_rotation(1.0, "head", [5, -10, 0])
    
    builder.set_joint_rotation(2.0, "shoulder_r", [0, 15, 0])
    builder.set_joint_rotation(2.0, "elbow_r", [0, 0, 90])
    builder.set_joint_rotation(2.0, "wrist_r", [0, 0, 30])
    builder.set_joint_rotation(2.0, "head", [5, 10, 0])
    
    builder.set_joint_rotation(3.0, "shoulder_r", [0, 15, 0])
    builder.set_joint_rotation(3.0, "elbow_r", [0, 0, 90])
    builder.set_joint_rotation(3.0, "wrist_r", [0, 0, 30])
    builder.set_joint_rotation(3.0, "head", [0, -5, 0])
    
    # Add context triggers
    builder.add_context_trigger("thinking")
    builder.add_context_trigger("contemplating")
    builder.add_context_trigger("considering")
    
    # Export the gesture
    gesture_path = os.path.join(args.output_dir, "thoughtful_pose.json")
    builder.export_gesture(gesture_path)
    print(f"Exported gesture to: {gesture_path}")
    
    # 2. Use a template to create a gesture
    print("\nCreating gesture from template:")
    builder.use_template("wave", "energetic_wave")
    
    # Modify some properties
    builder.set_duration(3.0)  # Make it longer
    
    # Add some extra context triggers
    builder.add_context_trigger("excited_greeting")
    builder.add_context_trigger("enthusiastic_hello")
    
    # Export the modified gesture
    gesture_path = os.path.join(args.output_dir, "energetic_wave.json")
    builder.export_gesture(gesture_path)
    print(f"Exported gesture to: {gesture_path}")
    
    # 3. Create a mirrored gesture
    print("\nCreating mirrored gesture:")
    mirrored = builder.mirror_gesture("wave_left_hand")
    
    # Export the mirrored gesture
    gesture_path = os.path.join(args.output_dir, "wave_left_hand.json")
    with open(gesture_path, 'w') as f:
        json.dump({builder.gesture_name: mirrored}, f, indent=2)
    print(f"Exported mirrored gesture to: {gesture_path}")
    
    # Add the custom gesture to the control system
    control_system.add_gesture("thoughtful_pose", builder.current_gesture)
    
    if args.preview or args.render:
        # Create a simple script to demonstrate the gesture
        script_path = os.path.join(args.output_dir, "gesture_demo.json")
        
        script = {
            "version": "1.0",
            "fps": 30,
            "avatar": args.avatar_path,
            "sections": [
                {
                    "type": "speech",
                    "start_time": 0.5,
                    "end_time": 4.5,
                    "text": "This is a demonstration of our custom thoughtful pose gesture."
                },
                {
                    "type": "gesture",
                    "start_time": 1.0,
                    "gesture": "thoughtful_pose"
                },
                {
                    "type": "camera",
                    "start_time": 0.0,
                    "action": "preset",
                    "preset": "three-quarter"
                },
                {
                    "type": "camera",
                    "start_time": 3.0,
                    "action": "preset",
                    "preset": "side",
                    "transition": 1.0
                }
            ]
        }
        
        # Save the script
        with open(script_path, 'w') as f:
            json.dump(script, f, indent=2)
        
        print(f"\nCreated gesture demo script: {script_path}")
        
        # Load and preview/render the script
        control_system.load_animation_script(script_path)
        
        if args.preview:
            demo_preview_animation(control_system, args)
        elif args.render:
            demo_render_animation(control_system, args)

def demo_camera_path(control_system, args):
    """Demonstrate camera path functionality."""
    print("\n--- Camera Path Demo ---\n")
    
    # Create some camera paths
    create_demo_camera_paths(control_system)
    
    # Show available camera paths
    print("Available camera paths:")
    for path_name in control_system.camera_paths:
        path = control_system.camera_paths[path_name]
        print(f"  - {path_name} (duration: {path.duration:.1f}s, {len(path.keyframes)} keyframes)")
    
    # Export camera library
    library_path = os.path.join(args.output_dir, "camera_library.json")
    control_system.export_camera_library(library_path)
    print(f"\nExported camera library to: {library_path}")
    
    if args.preview or args.render:
        # Create a script to demonstrate the camera paths
        script_path = os.path.join(args.output_dir, "camera_path_demo.json")
        
        script = {
            "version": "1.0",
            "fps": 30,
            "avatar": args.avatar_path,
            "sections": [
                {
                    "type": "speech",
                    "start_time": 0.5,
                    "end_time": 4.5,
                    "text": "Let me demonstrate our camera path system. First, an orbit around the avatar."
                },
                {
                    "type": "camera",
                    "start_time": 0.0,
                    "action": "preset",
                    "preset": "front"
                },
                {
                    "type": "camera",
                    "start_time": 5.0,
                    "action": "path",
                    "path_name": "orbit",
                    "duration": 8.0
                },
                {
                    "type": "speech",
                    "start_time": 5.5,
                    "end_time": 9.5,
                    "text": "This orbit path shows the avatar from all angles with smooth transitions."
                },
                {
                    "type": "speech",
                    "start_time": 13.5,
                    "end_time": 17.5,
                    "text": "Next, let's try a flyby path from one side to the other."
                },
                {
                    "type": "camera",
                    "start_time": 14.0,
                    "action": "path",
                    "path_name": "flyby",
                    "duration": 5.0
                },
                {
                    "type": "speech",
                    "start_time": 19.5,
                    "end_time": 23.5,
                    "text": "Finally, a dramatic reveal sequence with changing field of view."
                },
                {
                    "type": "camera",
                    "start_time": 20.0,
                    "action": "path",
                    "path_name": "dramatic_reveal",
                    "duration": 5.0
                },
                {
                    "type": "camera",
                    "start_time": 25.0,
                    "action": "preset",
                    "preset": "front",
                    "transition": 1.0
                },
                {
                    "type": "speech",
                    "start_time": 25.5,
                    "end_time": 29.5,
                    "text": "Camera paths can significantly enhance the visual presentation of your avatar."
                }
            ]
        }
        
        # Save the script
        with open(script_path, 'w') as f:
            json.dump(script, f, indent=2)
        
        print(f"\nCreated camera path demo script: {script_path}")
        
        # Load and preview/render the script
        control_system.load_animation_script(script_path)
        
        if args.preview:
            demo_preview_animation(control_system, args)
        elif args.render:
            demo_render_animation(control_system, args)

def create_demo_camera_paths(control_system):
    """Create demonstration camera paths."""
    # The standard paths (orbit, flyby, dramatic_reveal) are already created
    # in the CameraPathLibrary initialization
    
    # Create a custom switchback path
    switchback_path = CameraPath("switchback")
    
    # Add keyframes
    kf1 = CameraKeyframe(
        time=0.0,
        position=[-1.0, 0.2, 1.0],
        target=[0.0, 0.0, 0.0],
        fov=45.0,
        ease_type="ease_in_out"
    )
    
    kf2 = CameraKeyframe(
        time=1.0,
        position=[1.0, 0.2, 1.0],
        target=[0.0, 0.0, 0.0],
        fov=50.0,
        ease_type="ease_in_out"
    )
    
    kf3 = CameraKeyframe(
        time=2.0,
        position=[0.0, 0.2, 1.5],
        target=[0.0, 0.0, 0.0],
        fov=40.0,
        ease_type="ease_in_out"
    )
    
    kf4 = CameraKeyframe(
        time=3.0,
        position=[0.0, 0.7, 0.7],
        target=[0.0, 0.0, 0.0],
        fov=35.0,
        ease_type="ease_out"
    )
    
    kf5 = CameraKeyframe(
        time=4.0,
        position=[0.0, 0.0, 1.0],
        target=[0.0, 0.0, 0.0],
        fov=45.0,
        ease_type="ease_in"
    )
    
    switchback_path.add_keyframe(kf1)
    switchback_path.add_keyframe(kf2)
    switchback_path.add_keyframe(kf3)
    switchback_path.add_keyframe(kf4)
    switchback_path.add_keyframe(kf5)
    
    # Add the path to the system
    control_system.add_camera_path(switchback_path)
    
    # Create another path using the helper methods
    control_system.create_camera_path(
        name="close_orbit",
        path_type="orbit",
        radius=0.6,
        height=0.1,
        duration=6.0,
        num_keyframes=12
    )

def demo_add_environment_objects(control_system):
    """Add demo environment objects for interaction."""
    # Add a button object
    control_system.add_environment_object(
        object_id="virtual_button",
        position=[0.5, 0.0, 0.0],
        rotation=[0.0, 0.0, 0.0],
        scale=[0.1, 0.1, 0.1],
        model_path="models/button.obj"  # This would be a real path in production
    )
    
    # Add a table object
    control_system.add_environment_object(
        object_id="table",
        position=[0.0, -0.5, 0.0],
        rotation=[0.0, 0.0, 0.0],
        scale=[1.0, 0.05, 0.5],
        model_path="models/table.obj"  # This would be a real path in production
    )
    
    # Add interaction points
    control_system.add_interaction_point(
        point_id="button_press",
        position=[0.5, 0.0, 0.0],
        interaction_type="touch",
        linked_object="virtual_button"
    )
    
    control_system.add_interaction_point(
        point_id="table_surface",
        position=[0.0, -0.45, 0.0],
        interaction_type="place",
        linked_object="table"
    )

def demo_add_custom_gesture(control_system):
    """Add a custom gesture to the gesture library."""
    # Add a "thinking" gesture
    thinking_gesture = {
        "type": "hand",
        "description": "Thinking pose with hand on chin",
        "duration": 3.0,
        "joints": ["shoulder_r", "elbow_r", "wrist_r", "head"],
        "keyframes": [
            {"time": 0.0, "pose": {
                "shoulder_r": [0, 0, 0], 
                "elbow_r": [0, 0, 0], 
                "wrist_r": [0, 0, 0],
                "head": [0, 0, 0]
            }},
            {"time": 1.0, "pose": {
                "shoulder_r": [0, 15, 0], 
                "elbow_r": [0, 0, 90], 
                "wrist_r": [0, 0, 30],
                "head": [5, -10, 0]
            }},
            {"time": 2.0, "pose": {
                "shoulder_r": [0, 15, 0], 
                "elbow_r": [0, 0, 90], 
                "wrist_r": [0, 0, 30],
                "head": [5, 10, 0]
            }},
            {"time": 3.0, "pose": {
                "shoulder_r": [0, 0, 0], 
                "elbow_r": [0, 0, 0], 
                "wrist_r": [0, 0, 0],
                "head": [0, 0, 0]
            }}
        ],
        "context_triggers": ["thinking", "contemplating", "considering"]
    }
    
    control_system.add_gesture("thinking", thinking_gesture)

def demo_preview_animation(control_system, args):
    """
    Preview the animation in real-time (simulated for this demo).
    
    Args:
        control_system: The animation control system
        args: Command-line arguments
    """
    print("\n--- Real-time Animation Preview ---\n")
    print("Press Ctrl+C to stop the preview")
    
    try:
        frame_count = 0
        max_frames = args.frames if args.frames > 0 else control_system.total_frames
        
        start_time = time.time()
        
        while frame_count < max_frames:
            # Step the animation and get frame data
            frame_data = control_system.step_animation()
            
            # Simulate display (in a real implementation, this would render to screen)
            if frame_count % 5 == 0:  # Only print every 5 frames to reduce output
                print(f"Frame {frame_data['frame']} | "
                      f"Time: {frame_data['time']:.2f}s | "
                      f"Camera: {frame_data['camera']['position']} | "
                      f"FOV: {frame_data['camera']['fov']:.1f}° | "
                      f"Blendshapes: {len(frame_data['blendshapes'])} | "
                      f"Bones: {len(frame_data['bones'])}")
            
            # Simulate frame timing for real-time playback
            frame_time = 1.0 / control_system.fps
            elapsed = time.time() - start_time
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
            
            start_time = time.time()
            frame_count += 1
            
            # Break if we've gone through the whole animation
            if frame_count >= control_system.total_frames and args.frames <= 0:
                break
    
    except KeyboardInterrupt:
        print("\nPreview stopped by user")

def demo_render_animation(control_system, args):
    """
    Render the animation to a video file.
    
    Args:
        control_system: The animation control system
        args: Command-line arguments
    """
    print("\n--- Rendering Animation ---\n")
    
    # Define output path
    output_path = args.output
    if not output_path:
        output_path = os.path.join(args.output_dir, "rendered_animation.mp4")
    
    # Determine frame range
    start_frame = 0
    end_frame = None
    if args.frames > 0:
        end_frame = args.frames
    
    # Render the animation
    control_system.render_animation(
        output_path=output_path,
        start_frame=start_frame,
        end_frame=end_frame,
        resolution=(1920, 1080)
    )
    
    print(f"\nAnimation rendered to: {output_path}")

def main():
    """Main function to parse arguments and run the demo."""
    parser = argparse.ArgumentParser(
        description="Avatar Animation Control System Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--avatar", dest="avatar_path", type=str, required=True,
                        help="Path to the avatar model file")
    
    # Optional arguments
    parser.add_argument("--mode", type=str, 
                        choices=["standard", "emotion_markup", "gesture_builder", "camera_path"],
                        default="standard", help="Demo mode")
    parser.add_argument("--script", type=str, default=None,
                        help="Path to animation script file (JSON)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path for rendered output (for render mode)")
    parser.add_argument("--output-dir", dest="output_dir", type=str, 
                        default="output/animation_demo",
                        help="Directory for output files")
    parser.add_argument("--frames", type=int, default=0,
                        help="Number of frames to preview/render (0 for all)")
    parser.add_argument("--preview", action="store_true",
                        help="Preview animation")
    parser.add_argument("--render", action="store_true",
                        help="Render animation to video")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU usage (no GPU)")
    
    args = parser.parse_args()
    
    # Create output directory
    ensure_directory(args.output_dir)
    
    # Run the demo
    demo_animation_control(args)

if __name__ == "__main__":
    main() 