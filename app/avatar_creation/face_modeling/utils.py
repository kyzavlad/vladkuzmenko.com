import os
import numpy as np
import cv2
import torch
from typing import Dict, List, Tuple, Union, Optional

def ensure_directory(directory_path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Loaded image as a numpy array
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def save_image(image: np.ndarray, save_path: str) -> None:
    """
    Save an image to file.
    
    Args:
        image: Image as a numpy array
        save_path: Path where to save the image
    """
    ensure_directory(os.path.dirname(save_path))
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, image_bgr)

def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Preprocess an image for face modeling tasks.
    
    Args:
        image: Input image as a numpy array
        target_size: Target size for resizing
        
    Returns:
        Preprocessed image
    """
    # Resize image
    resized_image = cv2.resize(image, target_size)
    
    # Normalize pixel values to [0, 1]
    normalized_image = resized_image.astype(np.float32) / 255.0
    
    return normalized_image

def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a numpy image.
    
    Args:
        tensor: Input tensor of shape (C, H, W) or (B, C, H, W)
        
    Returns:
        Image as a numpy array
    """
    if tensor.ndim == 4:
        tensor = tensor[0]  # Take the first image in batch
    
    # Move to CPU if on GPU
    tensor = tensor.detach().cpu()
    
    # Rearrange from (C, H, W) to (H, W, C)
    image = tensor.permute(1, 2, 0).numpy()
    
    # Denormalize if in [0, 1] range
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    return image

def image_to_tensor(image: np.ndarray, add_batch_dim: bool = True) -> torch.Tensor:
    """
    Convert a numpy image to a PyTorch tensor.
    
    Args:
        image: Input image as a numpy array
        add_batch_dim: Whether to add a batch dimension
        
    Returns:
        Image as a PyTorch tensor
    """
    # Normalize if not already in [0, 1] range
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    # Rearrange from (H, W, C) to (C, H, W)
    tensor = torch.from_numpy(image).permute(2, 0, 1)
    
    # Add batch dimension if requested
    if add_batch_dim:
        tensor = tensor.unsqueeze(0)
    
    return tensor

def get_device() -> torch.device:
    """
    Get the appropriate device for PyTorch operations.
    
    Returns:
        PyTorch device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
