#!/usr/bin/env python3
"""
Avatar Animation Example

This script demonstrates the Animation Framework capabilities for avatar generation,
including the First Order Motion Model, facial landmark tracking, and
natural micro-expression synthesis.
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.avatar.face_modeling import FaceModeling
from app.services.avatar.animation_framework import AnimationFramework

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function for avatar animation example."""
    parser = argparse.ArgumentParser(description="Avatar Animation Example")
    parser.add_argument("--face-model", required=False, help="Path to 3D face model (optional)")
    parser.add_argument("--driving-video", required=True, help="Path to driving video")
    parser.add_argument("--output-dir", default="outputs", help="Output directory (default: outputs)")
    parser.add_argument("--generate-model", action="store_true", help="Generate a 3D face model from the first frame")
    parser.add_argument("--smooth", type=float, default=0.8, help="Temporal smoothness (0.0-1.0, default: 0.8)")
    parser.add_argument("--window", type=int, default=5, help="Smoothing window size (default: 5)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the AnimationFramework
    animation_framework = AnimationFramework({
        "output_dir": args.output_dir
    })
    
    # Check if we need to generate a face model first
    face_model_path = args.face_model
    if args.generate_model or not face_model_path:
        logger.info("Generating a 3D face model from the first frame of the driving video")
        face_model_path = generate_face_model(args.driving_video, args.output_dir)
    
    # Animate the avatar
    animate_avatar(
        animation_framework,
        face_model_path,
        args.driving_video,
        args.smooth,
        args.window
    )

def generate_face_model(video_path, output_dir):
    """Generate a 3D face model from a video."""
    logger.info(f"Extracting first frame from {video_path}")
    
    import cv2
    
    # Extract first frame from video
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise RuntimeError(f"Failed to extract frame from {video_path}")
    
    # Save the frame as an image
    image_path = os.path.join(output_dir, "first_frame.jpg")
    cv2.imwrite(image_path, frame)
    
    logger.info(f"First frame saved to {image_path}")
    
    # Initialize the FaceModeling service
    face_modeling = FaceModeling()
    
    # Generate a 3D face model from the image
    logger.info("Generating 3D face model...")
    
    options = {
        "detail_level": "high",
        "enable_texture_mapping": True,
        "enable_detail_refinement": True,
        "enable_identity_verification": True
    }
    
    try:
        result = face_modeling.generate_from_image(image_path, options)
        logger.info(f"Face model generated with ID: {result.model_id}")
        logger.info(f"Model path: {result.model_path}")
        logger.info(f"Quality score: {result.quality_score:.2f}")
        
        return result.model_path
    except Exception as e:
        logger.error(f"Error generating face model: {str(e)}")
        raise

def animate_avatar(animation_framework, face_model_path, driving_video_path, smoothness, window_size):
    """Animate an avatar using the First Order Motion Model."""
    logger.info(f"Animating avatar using model {face_model_path} and driving video {driving_video_path}")
    
    options = {
        "enhance_micro_expressions": True,
        "temporal_smoothness": smoothness,
        "smoothing_window": window_size
    }
    
    try:
        # Time the animation process
        start_time = time.time()
        
        # Animate the avatar
        result = animation_framework.animate_from_video(
            face_model_path=face_model_path,
            driving_video_path=driving_video_path,
            options=options
        )
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Print results
        logger.info(f"Animation completed in {elapsed_time:.2f} seconds")
        logger.info(f"Animation ID: {result.animation_id}")
        logger.info(f"Output path: {result.output_path}")
        logger.info(f"Duration: {result.duration:.2f} seconds")
        logger.info(f"Frame count: {result.frame_count}")
        logger.info(f"FPS: {result.fps:.2f}")
        
        # Display animation dimensions and format
        import cv2
        cap = cv2.VideoCapture(result.output_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        logger.info(f"Animation dimensions: {width}x{height}")
        logger.info(f"Animation format: {os.path.splitext(result.output_path)[1]}")
        
        logger.info(f"Animation saved to {result.output_path}")
        logger.info("You can view the animation using any video player")
        
    except Exception as e:
        logger.error(f"Error animating avatar: {str(e)}")
        raise

if __name__ == "__main__":
    main() 