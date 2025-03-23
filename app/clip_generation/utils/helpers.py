"""
Clip Generation Service - Helper Utilities

This module provides utility functions for the Clip Generation Service.
"""

import os
import re
import shutil
import json
import uuid
import logging
import subprocess
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_unique_id() -> str:
    """Generate a unique ID for tasks and clips."""
    return str(uuid.uuid4())


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to make it safe for file systems.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    sanitized = re.sub(r'[\\/*?:"<>|]', "", filename)
    
    # Replace spaces with underscores
    sanitized = re.sub(r'\s+', "_", sanitized)
    
    # Limit length
    if len(sanitized) > 255:
        base, ext = os.path.splitext(sanitized)
        sanitized = base[:255-len(ext)] + ext
    
    return sanitized


def format_timecode(seconds: float) -> str:
    """
    Format seconds as HH:MM:SS.mmm timecode for FFmpeg.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted timecode string
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"


def parse_timecode(timecode: str) -> float:
    """
    Parse a timecode string (HH:MM:SS.mmm) to seconds.
    
    Args:
        timecode: Timecode string
        
    Returns:
        Time in seconds
    """
    # Handle different formats
    if re.match(r'^\d+(\.\d+)?$', timecode):
        # Already in seconds
        return float(timecode)
    
    # HH:MM:SS.mmm format
    if re.match(r'^\d+:\d+:\d+(\.\d+)?$', timecode):
        parts = timecode.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    
    # MM:SS.mmm format
    if re.match(r'^\d+:\d+(\.\d+)?$', timecode):
        parts = timecode.split(':')
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    
    raise ValueError(f"Invalid timecode format: {timecode}")


def ensure_directory_exists(directory: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Created directory: {directory}")


def clear_directory(directory: str) -> None:
    """
    Clear all files from a directory.
    
    Args:
        directory: Directory path
    """
    if not os.path.exists(directory):
        return
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            os.unlink(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
    
    logger.debug(f"Cleared directory: {directory}")


def get_file_extension(path: str) -> str:
    """
    Get the file extension from a path.
    
    Args:
        path: File path
        
    Returns:
        File extension (with dot)
    """
    return os.path.splitext(path)[1].lower()


def get_file_size(path: str) -> int:
    """
    Get the size of a file in bytes.
    
    Args:
        path: File path
        
    Returns:
        File size in bytes
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    return os.path.getsize(path)


def human_readable_size(size_bytes: int) -> str:
    """
    Convert a file size in bytes to a human-readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Human-readable size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ("B", "KB", "MB", "GB", "TB", "PB")
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024
        i += 1
    
    return f"{size_bytes:.2f} {size_names[i]}"


def run_command(command: List[str], **kwargs) -> Tuple[int, str, str]:
    """
    Run a command and return the exit code, stdout, and stderr.
    
    Args:
        command: Command list
        **kwargs: Additional arguments for subprocess.run
        
    Returns:
        Tuple of (exit_code, stdout, stderr)
    """
    logger.debug(f"Running command: {' '.join(command)}")
    
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **kwargs
    )
    
    return result.returncode, result.stdout, result.stderr


def send_callback(url: str, data: Dict[str, Any]) -> bool:
    """
    Send a callback notification to a URL.
    
    Args:
        url: Callback URL
        data: Data to send
        
    Returns:
        True if successful, False otherwise
    """
    import requests
    
    try:
        response = requests.post(
            url,
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        return response.status_code >= 200 and response.status_code < 300
    except Exception as e:
        logger.error(f"Error sending callback to {url}: {str(e)}")
        return False


def calculate_clip_duration(start_time: float, end_time: float) -> float:
    """
    Calculate the duration of a clip.
    
    Args:
        start_time: Start time in seconds
        end_time: End time in seconds
        
    Returns:
        Duration in seconds
    """
    if end_time <= start_time:
        raise ValueError("End time must be greater than start time")
    
    return end_time - start_time 