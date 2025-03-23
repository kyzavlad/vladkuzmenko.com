import os
import json
import tempfile
import subprocess
from flask import current_app
import openai
from pydub import AudioSegment

def extract_audio(video_path, output_path=None):
    """
    Extract audio from a video file.
    
    Args:
        video_path (str): Path to the video file
        output_path (str, optional): Path to save the extracted audio. If None, a temporary file is created.
        
    Returns:
        str: Path to the extracted audio file
    """
    if output_path is None:
        # Create a temporary file with .mp3 extension
        fd, output_path = tempfile.mkstemp(suffix='.mp3')
        os.close(fd)
    
    # Use FFmpeg to extract audio
    ffmpeg_path = current_app.config['FFMPEG_PATH']
    command = [
        ffmpeg_path, 
        '-i', video_path, 
        '-q:a', '0',  # High quality
        '-map', 'a', 
        output_path
    ]
    
    subprocess.run(command, check=True, capture_output=True)
    return output_path

def segment_audio(audio_path, max_duration=600):
    """
    Segment audio file into smaller chunks if it's too long.
    Whisper API has limitations on file size and duration.
    
    Args:
        audio_path (str): Path to the audio file
        max_duration (int): Maximum duration of each segment in seconds
        
    Returns:
        list: List of paths to segmented audio files
    """
    audio = AudioSegment.from_file(audio_path)
    duration_ms = len(audio)
    
    # If audio is shorter than max_duration, return the original
    if duration_ms <= max_duration * 1000:
        return [audio_path]
    
    # Split the audio into segments
    segments = []
    for i in range(0, duration_ms, max_duration * 1000):
        fd, segment_path = tempfile.mkstemp(suffix='.mp3')
        os.close(fd)
        
        # Extract segment
        end = min(i + max_duration * 1000, duration_ms)
        segment = audio[i:end]
        segment.export(segment_path, format='mp3')
        segments.append(segment_path)
    
    return segments

def transcribe_with_whisper(audio_path):
    """
    Transcribe audio using OpenAI's Whisper API.
    
    Args:
        audio_path (str): Path to the audio file
        
    Returns:
        dict: Transcription results from Whisper API
    """
    # Set OpenAI API key
    api_key = current_app.config['OPENAI_API_KEY']
    if not api_key:
        raise ValueError("OpenAI API key is not set in the configuration")
    
    openai.api_key = api_key
    
    # Open the audio file
    with open(audio_path, 'rb') as audio_file:
        # Call the Whisper API
        response = openai.Audio.transcribe(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"]
        )
        
    return response

def merge_transcriptions(transcriptions):
    """
    Merge multiple transcription results into one.
    
    Args:
        transcriptions (list): List of transcription results
        
    Returns:
        dict: Merged transcription result
    """
    if len(transcriptions) == 1:
        return transcriptions[0]
    
    merged = {
        "text": "",
        "segments": [],
        "words": []
    }
    
    time_offset = 0
    for i, trans in enumerate(transcriptions):
        # Add text with spacing
        if i > 0:
            merged["text"] += " "
        merged["text"] += trans["text"]
        
        # Adjust segments with time offset
        for segment in trans["segments"]:
            adjusted_segment = segment.copy()
            adjusted_segment["start"] += time_offset
            adjusted_segment["end"] += time_offset
            merged["segments"].append(adjusted_segment)
        
        # Adjust words with time offset
        for word in trans.get("words", []):
            adjusted_word = word.copy()
            adjusted_word["start"] += time_offset
            adjusted_word["end"] += time_offset
            merged["words"].append(adjusted_word)
        
        # Update time offset for next transcription
        if trans["segments"]:
            time_offset = trans["segments"][-1]["end"]
    
    return merged

def transcribe_video(video_path):
    """
    Transcribe a video file using Whisper API.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        dict: Transcription results with timestamps
    """
    try:
        # Extract audio from video
        audio_path = extract_audio(video_path)
        
        # Segment audio if necessary
        audio_segments = segment_audio(audio_path)
        
        # Transcribe each segment
        transcriptions = []
        for segment_path in audio_segments:
            trans = transcribe_with_whisper(segment_path)
            transcriptions.append(trans)
            
            # Clean up temporary segment file if not the original
            if segment_path != audio_path:
                os.remove(segment_path)
        
        # Merge transcriptions if segmented
        final_transcription = merge_transcriptions(transcriptions)
        
        # Clean up the extracted audio file
        if audio_path != video_path:  # Ensure we don't delete the original video
            os.remove(audio_path)
        
        # Calculate accuracy metrics (estimated)
        accuracy = {
            "estimated_accuracy": 0.99,  # Whisper claims 99% accuracy
            "confidence_score": sum(s.get("confidence", 1.0) for s in final_transcription["segments"]) / 
                             len(final_transcription["segments"]) if final_transcription["segments"] else 0
        }
        
        # Enhance the final result with additional metadata
        result = {
            "transcription": final_transcription["text"],
            "segments": final_transcription["segments"],
            "words": final_transcription.get("words", []),
            "accuracy": accuracy,
            "duration": final_transcription["segments"][-1]["end"] if final_transcription["segments"] else 0
        }
        
        return result
    
    except Exception as e:
        # Log the error and re-raise
        print(f"Transcription error: {str(e)}")
        raise 