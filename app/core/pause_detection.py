import os
import json
import tempfile
import subprocess
import numpy as np
from flask import current_app
import ffmpeg
from pydub import AudioSegment
from pydub.silence import detect_silence

def get_video_info(video_path):
    """
    Get information about the video file using FFprobe.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        dict: Video information including duration, format, etc.
    """
    ffprobe_path = current_app.config['FFPROBE_PATH']
    
    # Run FFprobe to get video metadata
    command = [
        ffprobe_path,
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        video_path
    ]
    
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    video_info = json.loads(result.stdout)
    
    return video_info

def extract_audio_for_analysis(video_path):
    """
    Extract audio from video for silence detection.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        str: Path to the extracted audio file
    """
    # Create a temporary file with .wav extension for better silence detection
    fd, audio_path = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    
    # Use FFmpeg to extract audio
    ffmpeg_path = current_app.config['FFMPEG_PATH']
    command = [
        ffmpeg_path, 
        '-i', video_path, 
        '-q:a', '0',
        '-ac', '1',  # Convert to mono for easier analysis
        '-ar', '16000',  # 16kHz sample rate
        audio_path
    ]
    
    subprocess.run(command, check=True, capture_output=True)
    return audio_path

def find_pauses(audio_path, min_silence_len=500, silence_thresh=-40):
    """
    Detect silent segments in the audio.
    
    Args:
        audio_path (str): Path to the audio file
        min_silence_len (int): Minimum silence length in milliseconds
        silence_thresh (int): Silence threshold in dB
        
    Returns:
        list: List of silent segments as (start_ms, end_ms) tuples
    """
    audio = AudioSegment.from_file(audio_path)
    
    # Detect silent segments
    silent_segments = detect_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh
    )
    
    return silent_segments

def filter_pauses(silent_segments, min_pause=500, max_pause=2000):
    """
    Filter silent segments to keep only those that are pauses.
    
    Args:
        silent_segments (list): List of silent segments as (start_ms, end_ms) tuples
        min_pause (int): Minimum pause duration in milliseconds
        max_pause (int): Maximum pause duration in milliseconds
        
    Returns:
        list: Filtered list of pauses
    """
    pauses = []
    
    for start, end in silent_segments:
        duration = end - start
        if min_pause <= duration <= max_pause:
            pauses.append((start, end))
    
    return pauses

def create_edit_list(video_duration, pauses):
    """
    Create a list of segments to keep in the final video.
    
    Args:
        video_duration (float): Duration of the video in milliseconds
        pauses (list): List of pauses as (start_ms, end_ms) tuples
        
    Returns:
        list: List of segments to keep as (start_ms, end_ms) tuples
    """
    segments_to_keep = []
    last_end = 0
    
    for start, end in pauses:
        if start > last_end:
            segments_to_keep.append((last_end / 1000, start / 1000))  # Convert to seconds
        last_end = end
    
    # Add the final segment if needed
    if last_end < video_duration:
        segments_to_keep.append((last_end / 1000, video_duration / 1000))
    
    return segments_to_keep

def create_filter_complex(segments):
    """
    Create FFmpeg filter_complex string for concatenating segments.
    
    Args:
        segments (list): List of segments to keep as (start_s, end_s) tuples
        
    Returns:
        str: FFmpeg filter_complex string
    """
    parts = []
    for i, (start, end) in enumerate(segments):
        parts.append(f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{i}];")
        parts.append(f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}];")
    
    for i in range(len(segments)):
        parts.append(f"[v{i}][a{i}]")
    
    parts.append(f"concat=n={len(segments)}:v=1:a=1[outv][outa]")
    
    return "".join(parts)

def remove_pauses_with_ffmpeg(input_path, output_path, segments):
    """
    Remove pauses from video using FFmpeg.
    
    Args:
        input_path (str): Path to the input video
        output_path (str): Path to save the output video
        segments (list): List of segments to keep as (start_s, end_s) tuples
        
    Returns:
        bool: True if successful
    """
    if not segments:
        # No segments to process, just copy the file
        ffmpeg_path = current_app.config['FFMPEG_PATH']
        command = [
            ffmpeg_path,
            '-i', input_path,
            '-c', 'copy',
            output_path
        ]
        subprocess.run(command, check=True, capture_output=True)
        return True
    
    # Create filter complex for concatenating segments
    filter_complex = create_filter_complex(segments)
    
    # Run FFmpeg with the filter complex
    ffmpeg_path = current_app.config['FFMPEG_PATH']
    command = [
        ffmpeg_path,
        '-i', input_path,
        '-filter_complex', filter_complex,
        '-map', '[outv]',
        '-map', '[outa]',
        '-c:v', 'libx264',
        '-c:a', 'aac',
        output_path
    ]
    
    subprocess.run(command, check=True, capture_output=True)
    return True

def detect_and_remove_pauses(input_path, output_path, min_pause=0.5, max_pause=2.0):
    """
    Detect and remove pauses from a video.
    
    Args:
        input_path (str): Path to the input video
        output_path (str): Path to save the output video
        min_pause (float): Minimum pause duration in seconds
        max_pause (float): Maximum pause duration in seconds
        
    Returns:
        dict: Results including number of pauses removed and time saved
    """
    try:
        # Get video information
        video_info = get_video_info(input_path)
        video_duration_s = float(video_info['format']['duration'])
        video_duration_ms = video_duration_s * 1000
        
        # Extract audio for analysis
        audio_path = extract_audio_for_analysis(input_path)
        
        # Find silent segments
        silent_segments = find_pauses(
            audio_path, 
            min_silence_len=int(min_pause * 1000),  # Convert to ms
            silence_thresh=-40
        )
        
        # Filter to get only pauses
        pauses = filter_pauses(
            silent_segments,
            min_pause=int(min_pause * 1000),  # Convert to ms
            max_pause=int(max_pause * 1000)   # Convert to ms
        )
        
        # Calculate total time in pauses
        total_pause_time_ms = sum(end - start for start, end in pauses)
        
        # Create edit list
        segments_to_keep = create_edit_list(video_duration_ms, pauses)
        
        # Remove pauses from video
        remove_pauses_with_ffmpeg(input_path, output_path, segments_to_keep)
        
        # Clean up temporary audio file
        os.remove(audio_path)
        
        # Return results
        return {
            "pauses_removed": len(pauses),
            "time_saved": total_pause_time_ms / 1000,  # Convert to seconds
            "original_duration": video_duration_s,
            "new_duration": video_duration_s - (total_pause_time_ms / 1000)
        }
    
    except Exception as e:
        # Log the error and re-raise
        print(f"Pause detection error: {str(e)}")
        raise 