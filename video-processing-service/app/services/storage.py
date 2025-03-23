import os
import json
import subprocess
import shutil
import boto3
from typing import Dict, Any, Optional, BinaryIO
from pathlib import Path

from app.core.config import settings

class StorageException(Exception):
    """Base exception for storage service errors"""
    pass

def save_video(file_content: BinaryIO, filename: str) -> str:
    """
    Save a video file to the configured storage (local or S3)
    
    Args:
        file_content: File-like object containing the video data
        filename: Filename to save the video as
        
    Returns:
        str: The path or URL where the video is stored
        
    Raises:
        StorageException: If there's an error saving the file
    """
    if settings.STORAGE_TYPE == "local":
        return _save_local(file_content, filename)
    elif settings.STORAGE_TYPE == "s3":
        return _save_s3(file_content, filename)
    else:
        raise StorageException(f"Unsupported storage type: {settings.STORAGE_TYPE}")

def _save_local(file_content: BinaryIO, filename: str) -> str:
    """Save video to local storage"""
    # Ensure directory exists
    os.makedirs(settings.LOCAL_STORAGE_PATH, exist_ok=True)
    
    # Create full path
    file_path = os.path.join(settings.LOCAL_STORAGE_PATH, filename)
    
    try:
        # Write file
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file_content, f)
        return file_path
    except Exception as e:
        raise StorageException(f"Failed to save file locally: {str(e)}")

def _save_s3(file_content: BinaryIO, filename: str) -> str:
    """Save video to S3 storage"""
    if not all([settings.S3_BUCKET_NAME, settings.S3_ACCESS_KEY, settings.S3_SECRET_KEY]):
        raise StorageException("S3 storage is not properly configured")
    
    try:
        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.S3_ACCESS_KEY,
            aws_secret_access_key=settings.S3_SECRET_KEY,
            region_name=settings.S3_REGION
        )
        
        # Upload file to S3
        s3_client.upload_fileobj(
            file_content, 
            settings.S3_BUCKET_NAME, 
            filename
        )
        
        # Return S3 URL
        return f"s3://{settings.S3_BUCKET_NAME}/{filename}"
    except Exception as e:
        raise StorageException(f"Failed to save file to S3: {str(e)}")

def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    Extract metadata from a video file using FFmpeg
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dict: Dictionary containing video metadata (duration, width, height, fps, etc.)
        
    Raises:
        StorageException: If there's an error extracting metadata
    """
    try:
        # Get video metadata using ffprobe (part of FFmpeg)
        cmd = [
            settings.FFMPEG_PATH.replace('ffmpeg', 'ffprobe'),  # Use ffprobe
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        metadata = json.loads(result.stdout)
        
        # Extract relevant information
        video_info = {}
        
        # Get format information
        if 'format' in metadata:
            if 'duration' in metadata['format']:
                video_info['duration'] = float(metadata['format']['duration'])
            if 'size' in metadata['format']:
                video_info['file_size'] = int(metadata['format']['size'])
        
        # Find video stream
        if 'streams' in metadata:
            for stream in metadata['streams']:
                if stream.get('codec_type') == 'video':
                    # Found video stream
                    video_info['width'] = stream.get('width')
                    video_info['height'] = stream.get('height')
                    
                    # Calculate FPS
                    if 'r_frame_rate' in stream:
                        fps_parts = stream['r_frame_rate'].split('/')
                        if len(fps_parts) == 2 and int(fps_parts[1]) != 0:
                            video_info['fps'] = float(int(fps_parts[0]) / int(fps_parts[1]))
                    
                    # No need to look further once we found the video stream
                    break
        
        return video_info
    except subprocess.CalledProcessError as e:
        raise StorageException(f"Error running ffprobe: {e.stderr}")
    except json.JSONDecodeError:
        raise StorageException("Failed to parse ffprobe output")
    except Exception as e:
        raise StorageException(f"Error extracting video info: {str(e)}")

def delete_video(path: str) -> bool:
    """
    Delete a video file from storage
    
    Args:
        path: Path or URL to the video file
        
    Returns:
        bool: True if deletion was successful
        
    Raises:
        StorageException: If there's an error deleting the file
    """
    if path.startswith('s3://'):
        return _delete_s3(path)
    else:
        return _delete_local(path)

def _delete_local(path: str) -> bool:
    """Delete a file from local storage"""
    try:
        if os.path.exists(path):
            os.remove(path)
            return True
        return False
    except Exception as e:
        raise StorageException(f"Failed to delete local file: {str(e)}")

def _delete_s3(s3_url: str) -> bool:
    """Delete a file from S3 storage"""
    if not all([settings.S3_BUCKET_NAME, settings.S3_ACCESS_KEY, settings.S3_SECRET_KEY]):
        raise StorageException("S3 storage is not properly configured")
    
    try:
        # Parse S3 URL to get bucket and key
        parts = s3_url.replace('s3://', '').split('/')
        bucket = parts[0]
        key = '/'.join(parts[1:])
        
        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.S3_ACCESS_KEY,
            aws_secret_access_key=settings.S3_SECRET_KEY,
            region_name=settings.S3_REGION
        )
        
        # Delete file from S3
        s3_client.delete_object(Bucket=bucket, Key=key)
        return True
    except Exception as e:
        raise StorageException(f"Failed to delete file from S3: {str(e)}") 