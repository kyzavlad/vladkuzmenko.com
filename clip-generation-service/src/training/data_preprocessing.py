import os
import cv2
import numpy as np
from typing import List, Tuple, Dict
import json
import shutil
from tqdm import tqdm
import albumentations as A
from concurrent.futures import ThreadPoolExecutor
import logging

def create_augmentation_pipeline() -> A.Compose:
    """Create data augmentation pipeline."""
    return A.Compose([
        # Spatial augmentations
        A.RandomResizedCrop(
            height=224,
            width=224,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            p=0.5
        ),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),
        
        # Color augmentations
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0
            )
        ], p=0.5),
        
        # Noise and blur
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0)
        ], p=0.3),
        
        # Occlusion simulation
        A.CoarseDropout(
            max_holes=8,
            max_height=30,
            max_width=30,
            min_holes=2,
            min_height=10,
            min_width=10,
            fill_value=0,
            p=0.3
        )
    ], bbox_params=A.BboxParams(
        format="pascal_voc",
        min_area=100,
        min_visibility=0.5,
        label_fields=["class_labels"]
    ))

def process_video(
    video_path: str,
    output_dir: str,
    sequence_length: int = 16,
    stride: int = 8,
    min_face_size: Tuple[int, int] = (30, 30),
    augment: bool = True,
    num_augmentations: int = 3
) -> List[Dict]:
    """
    Process video and extract face tracking sequences.
    
    Args:
        video_path (str): Path to input video
        output_dir (str): Output directory for frames and annotations
        sequence_length (int): Length of extracted sequences
        stride (int): Stride between sequences
        min_face_size (Tuple[int, int]): Minimum face size
        augment (bool): Whether to apply augmentations
        num_augmentations (int): Number of augmented copies
        
    Returns:
        List[Dict]: Sequence annotations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    
    # Initialize augmentation pipeline
    if augment:
        transform = create_augmentation_pipeline()
    
    # Process frames
    frames = []
    annotations = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=min_face_size
        )
        
        # Save frame
        frame_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        # Save annotations
        frame_annos = []
        for i, (x, y, w, h) in enumerate(faces):
            frame_annos.append({
                "frame_idx": frame_idx,
                "bbox": [x, y, x + w, y + h],
                "identity_id": i  # Simple identity assignment
            })
        
        frames.append(frame)
        annotations.append(frame_annos)
        frame_idx += 1
    
    cap.release()
    
    # Extract sequences
    sequences = []
    for i in range(0, len(frames) - sequence_length + 1, stride):
        seq_frames = frames[i:i + sequence_length]
        seq_annos = annotations[i:i + sequence_length]
        
        # Create sequence annotation
        sequence = {
            "frames": [
                f"frame_{j:06d}.jpg"
                for j in range(i, i + sequence_length)
            ],
            "annotations": seq_annos
        }
        
        sequences.append(sequence)
        
        # Apply augmentations
        if augment:
            for aug_idx in range(num_augmentations):
                aug_frames = []
                aug_annos = []
                
                for frame, anno in zip(seq_frames, seq_annos):
                    # Prepare bounding boxes
                    bboxes = [a["bbox"] for a in anno]
                    class_labels = ["face"] * len(bboxes)
                    
                    # Apply augmentation
                    transformed = transform(
                        image=frame,
                        bboxes=bboxes,
                        class_labels=class_labels
                    )
                    
                    # Save augmented frame
                    frame_name = f"frame_{i:06d}_aug_{aug_idx:02d}.jpg"
                    frame_path = os.path.join(output_dir, frame_name)
                    cv2.imwrite(
                        frame_path,
                        cv2.cvtColor(transformed["image"], cv2.COLOR_RGB2BGR)
                    )
                    
                    # Update annotations
                    frame_annos = []
                    for bbox, identity in zip(
                        transformed["bboxes"],
                        [a["identity_id"] for a in anno]
                    ):
                        frame_annos.append({
                            "frame_idx": i,
                            "bbox": list(map(int, bbox)),
                            "identity_id": identity
                        })
                    
                    aug_frames.append(frame_name)
                    aug_annos.append(frame_annos)
                
                # Create augmented sequence annotation
                aug_sequence = {
                    "frames": aug_frames,
                    "annotations": aug_annos,
                    "augmented": True,
                    "augmentation_id": aug_idx
                }
                
                sequences.append(aug_sequence)
    
    return sequences

def preprocess_dataset(
    input_dir: str,
    output_dir: str,
    sequence_length: int = 16,
    stride: int = 8,
    min_face_size: Tuple[int, int] = (30, 30),
    augment: bool = True,
    num_augmentations: int = 3,
    num_workers: int = 4
):
    """
    Preprocess face tracking dataset.
    
    Args:
        input_dir (str): Input directory containing videos
        output_dir (str): Output directory for processed data
        sequence_length (int): Length of extracted sequences
        stride (int): Stride between sequences
        min_face_size (Tuple[int, int]): Minimum face size
        augment (bool): Whether to apply augmentations
        num_augmentations (int): Number of augmented copies
        num_workers (int): Number of worker threads
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all videos
    videos = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith((".mp4", ".avi", ".mov")):
                videos.append(os.path.join(root, file))
    
    logger.info(f"Found {len(videos)} videos")
    
    # Process videos in parallel
    all_sequences = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        
        for video_path in videos:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_output_dir = os.path.join(output_dir, video_name)
            
            future = executor.submit(
                process_video,
                video_path,
                video_output_dir,
                sequence_length,
                stride,
                min_face_size,
                augment,
                num_augmentations
            )
            futures.append((video_name, future))
        
        # Collect results
        for video_name, future in tqdm(futures):
            try:
                sequences = future.result()
                all_sequences.extend([
                    {**seq, "video_name": video_name}
                    for seq in sequences
                ])
            except Exception as e:
                logger.error(f"Error processing {video_name}: {str(e)}")
    
    # Save dataset metadata
    metadata = {
        "num_sequences": len(all_sequences),
        "sequence_length": sequence_length,
        "stride": stride,
        "min_face_size": min_face_size,
        "augmented": augment,
        "num_augmentations": num_augmentations
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save sequence annotations
    annotations_path = os.path.join(output_dir, "annotations.json")
    with open(annotations_path, "w") as f:
        json.dump(all_sequences, f, indent=2)
    
    logger.info(f"Processed {len(all_sequences)} sequences")
    logger.info(f"Metadata saved to: {metadata_path}")
    logger.info(f"Annotations saved to: {annotations_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sequence_length", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--min_face_width", type=int, default=30)
    parser.add_argument("--min_face_height", type=int, default=30)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--num_augmentations", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    preprocess_dataset(
        args.input_dir,
        args.output_dir,
        args.sequence_length,
        args.stride,
        (args.min_face_width, args.min_face_height),
        args.augment,
        args.num_augmentations,
        args.num_workers
    ) 