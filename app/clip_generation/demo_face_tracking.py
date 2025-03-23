#!/usr/bin/env python3
"""
Face Tracking System Demo

This script demonstrates the face tracking capabilities of the Clip Generation Service,
including multi-model face detection, face recognition, temporal tracking with Kalman filters,
and smart framing with speaker focus.

Usage:
    python demo_face_tracking.py --video_path <path> [options]

Options:
    --video_path PATH       Path to input video file
    --model_dir PATH        Path to model directory [default: models]
    --output_path PATH      Path to output video file [default: output.mp4]
    --display               Display video during processing
    --width WIDTH           Output width [default: 1280]
    --height HEIGHT         Output height [default: 720]
    --fps FPS               Output FPS [default: source video FPS]
    --detection_interval N  Run detection every N frames [default: 5]
    --smooth_factor N       Smoothing factor for camera movement (0-1) [default: 0.8]
    --save_debug            Save debug frames with detection visualization
    --debug_dir PATH        Directory to save debug frames [default: debug_frames]
    --device DEVICE         Device for inference (cpu, cuda) [default: cpu]
    --verbose               Enable verbose logging
"""

import os
import sys
import cv2
import numpy as np
import argparse
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm

# Import face tracking modules
from app.clip_generation.services.face_tracking import FaceBox
from app.clip_generation.services.face_tracking_manager import FaceTrackingManager, TrackedFace
from app.clip_generation.services.smart_framing import SmartFraming

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("face_tracking_demo")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Face Tracking Demo")
    
    # Input/output options
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to input video file")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to output video file (optional)")
    parser.add_argument("--display", action="store_true",
                        help="Display video with tracking")
    
    # Tracking options
    parser.add_argument("--detection_interval", type=int, default=5,
                        help="Run detection every N frames")
    parser.add_argument("--recognition_interval", type=int, default=15,
                        help="Run recognition every N frames")
    
    # Performance optimization options
    parser.add_argument("--sampling_strategy", type=str, default="uniform",
                        choices=["uniform", "adaptive", "keyframe", "motion"],
                        help="Frame sampling strategy")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for face detection")
    parser.add_argument("--worker_threads", type=int, default=2,
                        help="Number of worker threads for batch processing")
    parser.add_argument("--use_gpu", action="store_true",
                        help="Use GPU acceleration if available")
    
    # Framing options
    parser.add_argument("--width", type=int, default=1280,
                        help="Output width")
    parser.add_argument("--height", type=int, default=720,
                        help="Output height")
    parser.add_argument("--rule_of_thirds", action="store_true",
                        help="Apply rule of thirds composition")
    parser.add_argument("--smoothing", type=float, default=0.8,
                        help="Smoothing factor for camera movement (0-1)")
    
    # Debug options
    parser.add_argument("--debug", action="store_true",
                        help="Show debug information")
    parser.add_argument("--save_debug_frames", action="store_true",
                        help="Save debug frames")
    parser.add_argument("--debug_output_dir", type=str, default="debug_frames",
                        help="Directory to save debug frames")
    
    return parser.parse_args()


def draw_faces(
    frame: np.ndarray,
    tracked_faces: Dict[int, TrackedFace],
    speaker_id: Optional[int] = None
) -> np.ndarray:
    """
    Draw face boxes and information on the frame.
    
    Args:
        frame: Input frame
        tracked_faces: Dictionary of tracked faces
        speaker_id: ID of the current speaker
    
    Returns:
        Frame with drawn faces
    """
    result = frame.copy()
    height, width = result.shape[:2]
    
    # Draw each face
    for face_id, face in tracked_faces.items():
        box = face.box
        
        # Convert to pixel coordinates
        x1, y1 = int(box.x1), int(box.y1)
        x2, y2 = int(box.x2), int(box.y2)
        
        # Choose color based on speaker status
        color = (0, 255, 0)  # Green for regular faces
        thickness = 2
        
        if face_id == speaker_id:
            color = (0, 0, 255)  # Red for speaker
            thickness = 3
        
        # Draw bounding box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        
        # Draw face ID and info
        info_text = f"ID: {face_id}"
        
        # Add identity if available
        if face.identity is not None:
            info_text += f" | {face.identity.name}"
        
        # Add confidence
        avg_conf = face.avg_confidence()
        info_text += f" | Conf: {avg_conf:.2f}"
        
        # Add track length
        info_text += f" | Track: {face.track_length}"
        
        # Draw text background
        text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(
            result,
            (x1, y1 - text_size[1] - 10),
            (x1 + text_size[0] + 10, y1),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        cv2.putText(
            result,
            info_text,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        
        # Draw landmarks if available
        if box.landmarks is not None:
            for lm in box.landmarks:
                lm_x, lm_y = int(lm[0]), int(lm[1])
                cv2.circle(result, (lm_x, lm_y), 2, (255, 0, 0), -1)
    
    # Draw frame info
    cv2.putText(
        result,
        f"Faces: {len(tracked_faces)} | Speaker: {speaker_id if speaker_id is not None else 'None'}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )
    
    return result


def draw_framing_rect(
    frame: np.ndarray,
    rect: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (0, 255, 255)
) -> np.ndarray:
    """
    Draw framing rectangle on the frame.
    
    Args:
        frame: Input frame
        rect: Framing rectangle (x, y, width, height)
        color: Rectangle color
    
    Returns:
        Frame with drawn rectangle
    """
    result = frame.copy()
    x, y, w, h = rect
    
    # Draw rectangle
    cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
    
    # Draw rule of thirds grid
    third_w = w // 3
    third_h = h // 3
    
    # Vertical lines
    for i in range(1, 3):
        cv2.line(
            result,
            (x + i * third_w, y),
            (x + i * third_w, y + h),
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
    
    # Horizontal lines
    for i in range(1, 3):
        cv2.line(
            result,
            (x, y + i * third_h),
            (x + w, y + i * third_h),
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
    
    return result


def process_video(args):
    """Process a video with face tracking."""
    # Load video
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {args.video_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Video: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # Create output video writer if needed
    out = None
    if args.output_path:
        output_width = args.width
        output_height = args.height
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output_path, fourcc, fps, (output_width, output_height))
    
    # Create debug directory if needed
    if args.save_debug_frames:
        os.makedirs(args.debug_output_dir, exist_ok=True)
    
    # Initialize face tracking manager with performance optimization
    tracker = FaceTrackingManager(
        detection_interval=args.detection_interval,
        recognition_interval=args.recognition_interval,
        sampling_strategy=args.sampling_strategy,
        batch_size=args.batch_size,
        worker_threads=args.worker_threads,
        use_gpu=args.use_gpu
    )
    
    # Initialize smart framing
    framer = SmartFraming(
        target_width=args.width,
        target_height=args.height,
        smoothing_factor=args.smoothing,
        rule_of_thirds=args.rule_of_thirds
    )
    
    # Process video frames
    frame_count = 0
    processing_times = []
    
    try:
        progress_bar = tqdm(total=total_frames, desc="Processing video")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Measure processing time
            start_time = time.time()
            
            # Process frame
            tracked_faces = tracker.process_frame(frame)
            
            # Apply smart framing
            framed_image, framing_rect = framer.frame_image(
                frame, tracked_faces, tracker.speaker_id
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Draw tracking information
            if args.display or args.output_path:
                # Draw faces on the original frame
                debug_frame = frame.copy()
                debug_frame = draw_faces(debug_frame, tracked_faces, tracker.speaker_id)
                
                # Draw framing rectangle
                debug_frame = draw_framing_rect(debug_frame, framing_rect.to_tuple())
                
                # Draw metrics
                avg_time = sum(processing_times[-30:]) / min(len(processing_times), 30)
                fps_text = f"Processing: {1.0/avg_time:.1f} FPS"
                cv2.putText(debug_frame, fps_text, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Draw sampling info
                if args.sampling_strategy != "uniform":
                    if hasattr(tracker, 'stats'):
                        processed = tracker.stats.get("frames_processed", 0)
                        skipped = tracker.stats.get("frames_skipped", 0)
                        total = processed + skipped
                        if total > 0:
                            sample_rate = f"Sampling: {processed}/{total} frames ({processed/total*100:.1f}%)"
                            cv2.putText(debug_frame, sample_rate, (10, 60), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Draw face count
                face_count = len(tracked_faces)
                cv2.putText(debug_frame, f"Faces: {face_count}", (10, 90), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Draw strategy info
                strategy_text = f"Strategy: {args.sampling_strategy}"
                cv2.putText(debug_frame, strategy_text, (width - 300, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Draw batch processing info
                if args.worker_threads > 0:
                    batch_text = f"Batch: {args.batch_size}x{args.worker_threads} workers"
                    cv2.putText(debug_frame, batch_text, (width - 300, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # GPU info
                gpu_text = f"GPU: {'On' if args.use_gpu else 'Off'}"
                cv2.putText(debug_frame, gpu_text, (width - 300, 90), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display video
            if args.display:
                # Show the original frame with tracking and the framed output
                combined = np.hstack((
                    cv2.resize(debug_frame, (640, 360)),
                    cv2.resize(framed_image, (640, 360))
                ))
                cv2.imshow('Face Tracking Demo', combined)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Write output video
            if out:
                out.write(framed_image)
            
            # Save debug frames
            if args.save_debug_frames and frame_count % 10 == 0:
                debug_path = os.path.join(args.debug_output_dir, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(debug_path, debug_frame)
                
                framed_path = os.path.join(args.debug_output_dir, f"framed_{frame_count:04d}.jpg")
                cv2.imwrite(framed_path, framed_image)
                
            frame_count += 1
            progress_bar.update(1)
    
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Print processing statistics
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            logger.info(f"Average processing time: {avg_time:.3f}s ({1.0/avg_time:.1f} FPS)")
            
            if hasattr(tracker, 'stats'):
                processed = tracker.stats.get("frames_processed", 0)
                skipped = tracker.stats.get("frames_skipped", 0)
                logger.info(f"Frames processed: {processed}/{frame_count} ({processed/frame_count*100:.1f}%)")
                logger.info(f"Frames skipped: {skipped}/{frame_count} ({skipped/frame_count*100:.1f}%)")
                
                if processed > 0 and "detection_time" in tracker.stats:
                    avg_detection = sum(tracker.stats["detection_time"]) / len(tracker.stats["detection_time"])
                    logger.info(f"Average detection time: {avg_detection:.3f}s")
                
                if "faces_detected" in tracker.stats and tracker.stats["faces_detected"]:
                    avg_faces = sum(tracker.stats["faces_detected"]) / len(tracker.stats["faces_detected"])
                    logger.info(f"Average faces detected: {avg_faces:.1f}")
        
        # Clean up
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Stop batch processing
        tracker.reset()
        
        logger.info("Processing complete")


def main():
    """Main entry point."""
    args = parse_args()
    
    logger.info("Starting Face Tracking Demo")
    logger.info(f"Input video: {args.video_path}")
    logger.info(f"Output video: {args.output_path}")
    
    success = process_video(args)
    
    if success:
        logger.info("Face tracking demo completed successfully")
        return 0
    else:
        logger.error("Face tracking demo failed")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 