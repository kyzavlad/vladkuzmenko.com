import os
import json
import time
import tempfile
import logging
from typing import Dict, List, Any, Optional, BinaryIO, Tuple
import openai
import requests
from datetime import datetime
import uuid
import httpx

from app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Configure OpenAI API key
openai.api_key = settings.OPENAI_API_KEY

class WhisperException(Exception):
    """Exception raised for errors in the Whisper transcription service"""
    pass

async def transcribe_audio(
    audio_file_path: str, 
    language: Optional[str] = None,
    word_timestamps: bool = True,
    prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Transcribe audio file using OpenAI Whisper API
    
    Args:
        audio_file_path: Path to the audio file to transcribe
        language: ISO-639-1 language code (e.g., 'en', 'es', etc.)
        word_timestamps: Whether to include word-level timestamps
        prompt: Optional prompt to guide the transcription
        
    Returns:
        Dict containing transcription results including:
        - text: The full transcription text
        - segments: List of timestamped segments
        - words: List of timestamped words (if word_timestamps=True)
        
    Raises:
        WhisperException: If transcription fails
    """
    try:
        # Validate file exists
        if not os.path.exists(audio_file_path):
            raise WhisperException(f"Audio file not found: {audio_file_path}")
        
        # Validate file size
        file_size_mb = os.path.getsize(audio_file_path) / (1024 * 1024)
        if file_size_mb > settings.MAX_AUDIO_SIZE_MB:
            raise WhisperException(
                f"Audio file too large: {file_size_mb:.2f}MB (max: {settings.MAX_AUDIO_SIZE_MB}MB)"
            )
        
        # Prepare API parameters
        api_params = {
            "model": settings.WHISPER_MODEL,
            "response_format": "verbose_json",
        }
        
        if language:
            api_params["language"] = language
            
        if prompt:
            api_params["prompt"] = prompt
            
        if word_timestamps:
            api_params["timestamp_granularities"] = ["word"]
        
        logger.info(f"Starting transcription of {os.path.basename(audio_file_path)}")
        
        # Measure transcription time
        start_time = time.time()
        
        # Open the file and send to OpenAI
        with open(audio_file_path, "rb") as audio_file:
            # Use OpenAI's async client with httpx
            async with httpx.AsyncClient(timeout=settings.TRANSCRIPTION_TIMEOUT_SECONDS) as client:
                # Create form data
                files = {'file': (os.path.basename(audio_file_path), audio_file, 'audio/mpeg')}
                form_data = {}
                for key, value in api_params.items():
                    if isinstance(value, list):
                        # Handle lists like timestamp_granularities
                        for item in value:
                            form_data[f"{key}[]"] = item
                    else:
                        form_data[key] = value
                
                # Make API call
                headers = {"Authorization": f"Bearer {openai.api_key}"}
                response = await client.post(
                    "https://api.openai.com/v1/audio/transcriptions", 
                    files=files,
                    data=form_data,
                    headers=headers
                )
                
                if response.status_code != 200:
                    raise WhisperException(f"OpenAI API error: {response.text}")
                
                transcription = response.json()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info(f"Transcription completed in {processing_time:.2f} seconds")
        
        # Process and format the response
        result = {
            "text": transcription.get("text", ""),
            "segments": transcription.get("segments", []),
            "language": transcription.get("language", language),
            "duration": transcription.get("duration", None),
            "processing_time": processing_time,
        }
        
        # Include words if requested and available
        if word_timestamps and "words" in transcription:
            result["words"] = transcription["words"]
        
        return result
        
    except openai.OpenAIError as e:
        raise WhisperException(f"OpenAI API error: {str(e)}")
    except Exception as e:
        raise WhisperException(f"Transcription failed: {str(e)}")

def format_transcription_as_srt(segments: List[Dict[str, Any]]) -> str:
    """
    Convert transcription segments to SRT subtitle format
    
    Args:
        segments: List of transcription segments with start/end times
        
    Returns:
        str: Formatted SRT subtitle content
    """
    srt_content = ""
    
    for i, segment in enumerate(segments):
        # Get start and end times in seconds
        start_time = segment.get("start", 0)
        end_time = segment.get("end", 0)
        
        # Convert to SRT time format (HH:MM:SS,mmm)
        start_formatted = _format_time_srt(start_time)
        end_formatted = _format_time_srt(end_time)
        
        # Get the text content
        text = segment.get("text", "").strip()
        
        # Add the segment to SRT content
        srt_content += f"{i+1}\n"
        srt_content += f"{start_formatted} --> {end_formatted}\n"
        srt_content += f"{text}\n\n"
    
    return srt_content

def format_transcription_as_vtt(segments: List[Dict[str, Any]]) -> str:
    """
    Convert transcription segments to WebVTT subtitle format
    
    Args:
        segments: List of transcription segments with start/end times
        
    Returns:
        str: Formatted WebVTT subtitle content
    """
    vtt_content = "WEBVTT\n\n"
    
    for i, segment in enumerate(segments):
        # Get start and end times in seconds
        start_time = segment.get("start", 0)
        end_time = segment.get("end", 0)
        
        # Convert to VTT time format (HH:MM:SS.mmm)
        start_formatted = _format_time_vtt(start_time)
        end_formatted = _format_time_vtt(end_time)
        
        # Get the text content
        text = segment.get("text", "").strip()
        
        # Add the segment to VTT content
        vtt_content += f"{start_formatted} --> {end_formatted}\n"
        vtt_content += f"{text}\n\n"
    
    return vtt_content

def _format_time_srt(seconds: float) -> str:
    """Format time in seconds to SRT format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

def _format_time_vtt(seconds: float) -> str:
    """Format time in seconds to WebVTT format (HH:MM:SS.mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}.{milliseconds:03d}"

def extract_audio_from_video(video_path: str, output_path: Optional[str] = None) -> str:
    """
    Extract audio track from video file using FFmpeg
    
    Args:
        video_path: Path to the video file
        output_path: Optional path for the extracted audio file
        
    Returns:
        str: Path to the extracted audio file
        
    Raises:
        WhisperException: If audio extraction fails
    """
    try:
        import subprocess
        
        # If output path not specified, create a temporary file
        if not output_path:
            # Use the same directory as the video file
            video_dir = os.path.dirname(video_path)
            output_path = os.path.join(
                video_dir, 
                f"audio_{uuid.uuid4().hex}.mp3"
            )
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Use FFmpeg to extract audio
        command = [
            "ffmpeg",
            "-i", video_path,
            "-q:a", "0",           # Best quality
            "-map", "a",           # Extract only audio
            "-c:a", "libmp3lame",  # MP3 encoding
            "-y",                  # Overwrite output file if exists
            output_path
        ]
        
        # Run FFmpeg command
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        return output_path
        
    except subprocess.CalledProcessError as e:
        raise WhisperException(f"Failed to extract audio: {e.stderr}")
    except Exception as e:
        raise WhisperException(f"Audio extraction error: {str(e)}") 