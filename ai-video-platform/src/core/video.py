import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import torch
import torchvision.transforms as transforms
from PIL import Image
import ffmpeg
import os
import io
import tempfile

class VideoProcessor:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv']
    
    def extract_frames(
        self,
        video_path: str,
        start_time: float = 0,
        end_time: Optional[float] = None,
        fps: Optional[int] = None
    ) -> List[np.ndarray]:
        """Extract frames from video file."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / video_fps
        
        # Set frame extraction parameters
        if fps is None:
            fps = video_fps
        if end_time is None:
            end_time = duration
        
        # Calculate frame indices
        start_frame = int(start_time * video_fps)
        end_frame = int(end_time * video_fps)
        frame_interval = int(video_fps / fps)
        
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or cap.get(cv2.CAP_PROP_POS_FRAMES) > end_frame:
                break
            
            if len(frames) % frame_interval == 0:
                frames.append(frame)
        
        cap.release()
        return frames
    
    def process_frames(
        self,
        frames: List[np.ndarray],
        batch_size: int = 32
    ) -> List[np.ndarray]:
        """Process frames in batches using PyTorch."""
        processed_frames = []
        
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            batch_tensors = []
            
            for frame in batch:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                # Apply transforms
                tensor = self.transform(pil_image)
                batch_tensors.append(tensor)
            
            # Stack tensors and move to device
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Process batch (placeholder for actual processing)
            processed_batch = self._process_batch(batch_tensor)
            
            # Convert back to numpy arrays
            for tensor in processed_batch:
                # Denormalize and convert to numpy
                tensor = tensor.cpu().numpy()
                tensor = np.transpose(tensor, (1, 2, 0))
                tensor = (tensor * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
                tensor = tensor.astype(np.uint8)
                processed_frames.append(tensor)
        
        return processed_frames
    
    def _process_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Process a batch of frames (placeholder for actual processing)."""
        # Implement actual processing logic here
        return batch
    
    def save_video(
        self,
        frames: List[np.ndarray],
        output_path: str,
        fps: int = 30,
        codec: str = "libx264",
        audio_path: Optional[str] = None
    ):
        """Save frames as video file."""
        if not frames:
            raise ValueError("No frames to save")
        
        # Get frame dimensions
        height, width = frames[0].shape[:2]
        
        # Create temporary directory for frames
        temp_dir = "temp_frames"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save frames as images
        frame_paths = []
        for i, frame in enumerate(frames):
            frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
        
        # Prepare ffmpeg command
        stream = ffmpeg.input("temp_frames/frame_%06d.png", pattern_type="sequence", framerate=fps)
        
        if audio_path:
            audio = ffmpeg.input(audio_path)
            stream = ffmpeg.output(
                stream,
                audio,
                output_path,
                vcodec=codec,
                acodec="aac",
                video_bitrate="5000k",
                audio_bitrate="192k"
            )
        else:
            stream = ffmpeg.output(
                stream,
                output_path,
                vcodec=codec,
                video_bitrate="5000k"
            )
        
        # Run ffmpeg command
        ffmpeg.run(stream, overwrite_output=True)
        
        # Clean up temporary files
        for path in frame_paths:
            os.remove(path)
        os.rmdir(temp_dir)
    
    def analyze_video_quality(
        self,
        video_path: str
    ) -> dict:
        """Analyze video quality metrics."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        # Calculate quality metrics
        metrics = {
            "resolution": f"{width}x{height}",
            "fps": fps,
            "duration": duration,
            "total_frames": total_frames,
            "aspect_ratio": width / height,
            "bitrate": self._estimate_bitrate(video_path),
            "quality_score": self._calculate_quality_score(cap)
        }
        
        cap.release()
        return metrics
    
    def _estimate_bitrate(self, video_path: str) -> float:
        """Estimate video bitrate."""
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        return float(video_info.get('bit_rate', 0)) / 1000  # Convert to kbps
    
    def _calculate_quality_score(self, cap: cv2.VideoCapture) -> float:
        """Calculate overall quality score."""
        # Implement quality scoring logic here
        # This is a placeholder implementation
        return 0.85
    
    def load_video(
        self,
        video_path: str,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Load video frames and metadata."""
        cap = cv2.VideoCapture(video_path)
        
        # Get video metadata
        metadata = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        # Set frame range
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = metadata["frame_count"]
        
        # Read frames
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        while len(frames) < (end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        return frames, metadata
    
    def resize_video(
        self,
        frames: List[np.ndarray],
        target_size: Tuple[int, int],
        maintain_aspect: bool = True
    ) -> List[np.ndarray]:
        """Resize video frames."""
        resized_frames = []
        
        for frame in frames:
            if maintain_aspect:
                h, w = frame.shape[:2]
                aspect = w / h
                target_w, target_h = target_size
                target_aspect = target_w / target_h
                
                if aspect > target_aspect:
                    new_w = target_w
                    new_h = int(new_w / aspect)
                else:
                    new_h = target_h
                    new_w = int(new_h * aspect)
                
                resized = cv2.resize(frame, (new_w, new_h))
                
                # Create black background
                result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                
                # Center the resized frame
                y_offset = (target_h - new_h) // 2
                x_offset = (target_w - new_w) // 2
                result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            else:
                result = cv2.resize(frame, target_size)
            
            resized_frames.append(result)
        
        return resized_frames
    
    def apply_effects(
        self,
        frames: List[np.ndarray],
        effects: List[Dict[str, Any]]
    ) -> List[np.ndarray]:
        """Apply visual effects to frames."""
        processed_frames = frames.copy()
        
        for effect in effects:
            effect_type = effect.get("type")
            params = effect.get("params", {})
            
            if effect_type == "blur":
                kernel_size = params.get("kernel_size", 3)
                processed_frames = [
                    cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
                    for frame in processed_frames
                ]
            
            elif effect_type == "brightness":
                alpha = params.get("alpha", 1.0)
                beta = params.get("beta", 0)
                processed_frames = [
                    cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
                    for frame in processed_frames
                ]
            
            elif effect_type == "contrast":
                alpha = params.get("alpha", 1.0)
                processed_frames = [
                    cv2.convertScaleAbs(frame, alpha=alpha)
                    for frame in processed_frames
                ]
            
            elif effect_type == "color_balance":
                r = params.get("r", 1.0)
                g = params.get("g", 1.0)
                b = params.get("b", 1.0)
                processed_frames = [
                    cv2.convertScaleAbs(frame, alpha=1.0, beta=0)
                    for frame in processed_frames
                ]
                for i, frame in enumerate(processed_frames):
                    b, g, r = cv2.split(frame)
                    b = cv2.multiply(b, b)
                    g = cv2.multiply(g, g)
                    r = cv2.multiply(r, r)
                    processed_frames[i] = cv2.merge([b, g, r])
        
        return processed_frames
    
    def create_transition(
        self,
        frames1: List[np.ndarray],
        frames2: List[np.ndarray],
        transition_type: str = "fade",
        duration: int = 30
    ) -> List[np.ndarray]:
        """Create transition between two video segments."""
        if transition_type == "fade":
            transition_frames = []
            for i in range(duration):
                alpha = i / duration
                frame = cv2.addWeighted(
                    frames1[-1],
                    1 - alpha,
                    frames2[0],
                    alpha,
                    0
                )
                transition_frames.append(frame)
            return frames1[:-1] + transition_frames + frames2[1:]
        
        elif transition_type == "slide":
            transition_frames = []
            width = frames1[0].shape[1]
            for i in range(duration):
                offset = int((width * i) / duration)
                frame = np.zeros_like(frames1[0])
                frame[:, :width-offset] = frames1[-1][:, offset:]
                frame[:, width-offset:] = frames2[0][:, :offset]
                transition_frames.append(frame)
            return frames1[:-1] + transition_frames + frames2[1:]
        
        else:
            raise ValueError(f"Unsupported transition type: {transition_type}")
    
    def extract_audio(
        self,
        video_path: str,
        output_path: Optional[str] = None
    ) -> str:
        """Extract audio from video."""
        if output_path is None:
            output_path = tempfile.mktemp(suffix='.wav')
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create temporary video with audio
        temp_video = tempfile.mktemp(suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        cap.release()
        out.release()
        
        # Extract audio using ffmpeg
        os.system(f'ffmpeg -i {temp_video} -vn -acodec pcm_s16le -ar 44100 -ac 2 {output_path}')
        os.remove(temp_video)
        
        return output_path
    
    def merge_audio_video(
        self,
        video_path: str,
        audio_path: str,
        output_path: str
    ) -> None:
        """Merge audio and video streams."""
        os.system(f'ffmpeg -i {video_path} -i {audio_path} -c:v copy -c:a aac {output_path}')
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get detailed video information."""
        cap = cv2.VideoCapture(video_path)
        
        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
            "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
            "format": os.path.splitext(video_path)[1].lower()
        }
        
        cap.release()
        return info 