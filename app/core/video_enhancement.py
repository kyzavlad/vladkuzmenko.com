import os
import subprocess
from flask import current_app
import json
import tempfile

def enhance_video(input_path, output_path, enhancement_options=None):
    """
    Enhance video quality and audio clarity.
    
    Args:
        input_path (str): Path to the input video
        output_path (str): Path to save the output video
        enhancement_options (dict, optional): Enhancement options
        
    Returns:
        dict: Result information
    """
    try:
        # Default enhancement options
        default_options = {
            "denoise_video": True,
            "denoise_audio": True,
            "enhance_voice": True,
            "sharpen_video": True,
            "normalize_audio": True,
            "video_quality": "medium",  # Options: low, medium, high
            "audio_quality": "medium"   # Options: low, medium, high
        }
        
        # Use provided options or defaults
        options = {**default_options, **(enhancement_options or {})}
        
        # Prepare FFmpeg filters
        video_filters = []
        audio_filters = []
        
        # Apply video enhancements
        if options["denoise_video"]:
            # Add video denoising filter
            # Low, medium, high settings for noise reduction strength
            if options["video_quality"] == "low":
                video_filters.append("hqdn3d=1.5:1.5:6:6")  # Light denoise
            elif options["video_quality"] == "medium":
                video_filters.append("hqdn3d=3:3:6:6")  # Medium denoise
            else:  # high
                video_filters.append("hqdn3d=7:7:6:6")  # Strong denoise
        
        if options["sharpen_video"]:
            # Add sharpening filter
            # Different sharpening based on quality
            if options["video_quality"] == "low":
                video_filters.append("unsharp=3:3:0.5:3:3:0.5")  # Light sharpen
            elif options["video_quality"] == "medium":
                video_filters.append("unsharp=5:5:1.0:5:5:0.0")  # Medium sharpen
            else:  # high
                video_filters.append("unsharp=7:7:1.5:7:7:0.0")  # Strong sharpen
        
        # Apply audio enhancements
        if options["denoise_audio"]:
            # Add audio noise reduction
            # The strength depends on audio quality setting
            if options["audio_quality"] == "low":
                audio_filters.append("afftdn=nf=-25")  # Light noise reduction
            elif options["audio_quality"] == "medium":
                audio_filters.append("afftdn=nf=-30")  # Medium noise reduction
            else:  # high
                audio_filters.append("afftdn=nf=-40")  # Strong noise reduction
        
        if options["enhance_voice"]:
            # Add voice enhancement filters
            # Highpass to remove low rumble, lowpass to remove high hiss
            # Compression to make voice more consistent
            audio_filters.append("highpass=f=80,lowpass=f=12000")  # Basic frequency band for voice
            
            # Apply compression for voice clarity - different based on quality
            if options["audio_quality"] == "low":
                audio_filters.append("compand=0.3|0.3:1|1:-90/-60|-60/-40|-40/-30|-20/-20:6:0:-90:0.2")  # Light
            elif options["audio_quality"] == "medium":
                audio_filters.append("compand=0.3|0.3:1|1:-90/-60|-60/-40|-40/-30|-20/-15:6:0:-90:0.2")  # Medium
            else:  # high
                audio_filters.append("compand=0.2|0.2:1|1:-90/-60|-60/-40|-40/-30|-20/-10:6:0:-90:0.2")  # Strong
        
        if options["normalize_audio"]:
            # Add audio normalization filter
            # Set the volume to a target level based on quality
            if options["audio_quality"] == "low":
                audio_filters.append("dynaudnorm=f=200:g=15:r=0.8")  # Light normalization
            elif options["audio_quality"] == "medium":
                audio_filters.append("dynaudnorm=f=150:g=15:r=0.5")  # Medium normalization
            else:  # high
                audio_filters.append("dynaudnorm=f=100:g=15:r=0.3")  # Strong normalization
        
        # Build the FFmpeg command
        ffmpeg_path = current_app.config['FFMPEG_PATH']
        command = [ffmpeg_path, '-i', input_path]
        
        # Add video filters if any
        if video_filters:
            command.extend(['-vf', ','.join(video_filters)])
        
        # Add audio filters if any
        if audio_filters:
            command.extend(['-af', ','.join(audio_filters)])
        
        # Set encoding parameters based on quality
        if options["video_quality"] == "low":
            command.extend(['-c:v', 'libx264', '-crf', '28', '-preset', 'fast'])
        elif options["video_quality"] == "medium":
            command.extend(['-c:v', 'libx264', '-crf', '23', '-preset', 'medium'])
        else:  # high
            command.extend(['-c:v', 'libx264', '-crf', '18', '-preset', 'slow'])
        
        # Set audio encoding parameters
        if options["audio_quality"] == "low":
            command.extend(['-c:a', 'aac', '-b:a', '96k'])
        elif options["audio_quality"] == "medium":
            command.extend(['-c:a', 'aac', '-b:a', '128k'])
        else:  # high
            command.extend(['-c:a', 'aac', '-b:a', '192k'])
        
        # Add output path
        command.append(output_path)
        
        # Run the FFmpeg command
        subprocess.run(command, check=True, capture_output=True)
        
        # Return enhancement details
        enhancements_applied = []
        if options["denoise_video"]:
            enhancements_applied.append("Video noise reduction")
        if options["sharpen_video"]:
            enhancements_applied.append("Video sharpening")
        if options["denoise_audio"]:
            enhancements_applied.append("Audio noise reduction")
        if options["enhance_voice"]:
            enhancements_applied.append("Voice clarity enhancement")
        if options["normalize_audio"]:
            enhancements_applied.append("Audio normalization")
        
        return {
            "enhancements_applied": enhancements_applied,
            "video_quality": options["video_quality"],
            "audio_quality": options["audio_quality"]
        }
    
    except Exception as e:
        # Log the error and re-raise
        print(f"Video enhancement error: {str(e)}")
        raise

def extract_voice_track(input_path, output_path=None):
    """
    Extract and enhance the voice track from a video.
    
    Args:
        input_path (str): Path to the input video
        output_path (str, optional): Path to save the voice audio
        
    Returns:
        str: Path to the extracted voice audio
    """
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix='.aac')
        os.close(fd)
    
    # FFmpeg command to extract and enhance voice
    ffmpeg_path = current_app.config['FFMPEG_PATH']
    command = [
        ffmpeg_path,
        '-i', input_path,
        '-af', 'highpass=f=80,lowpass=f=12000,afftdn=nf=-30,compand=0.3|0.3:1|1:-90/-60|-60/-40|-40/-30|-20/-15:6:0:-90:0.2,dynaudnorm',
        '-vn',  # No video
        '-c:a', 'aac',
        '-b:a', '192k',
        output_path
    ]
    
    subprocess.run(command, check=True, capture_output=True)
    return output_path

def enhance_specific_video_segment(input_path, output_path, start_time, end_time, enhancement_options=None):
    """
    Enhance a specific segment of a video.
    
    Args:
        input_path (str): Path to the input video
        output_path (str): Path to save the output video
        start_time (float): Start time in seconds
        end_time (float): End time in seconds
        enhancement_options (dict, optional): Enhancement options
        
    Returns:
        dict: Result information
    """
    try:
        # Create temporary files for the segment
        fd, segment_path = tempfile.mkstemp(suffix='.mp4')
        os.close(fd)
        
        # Extract the segment
        ffmpeg_path = current_app.config['FFMPEG_PATH']
        extract_command = [
            ffmpeg_path,
            '-i', input_path,
            '-ss', str(start_time),
            '-to', str(end_time),
            '-c', 'copy',
            segment_path
        ]
        
        subprocess.run(extract_command, check=True, capture_output=True)
        
        # Enhance the segment
        enhance_result = enhance_video(segment_path, output_path, enhancement_options)
        
        # Clean up
        os.remove(segment_path)
        
        return enhance_result
    
    except Exception as e:
        # Log the error and re-raise
        print(f"Segment enhancement error: {str(e)}")
        raise

def optimize_for_target_aspect_ratio(input_path, output_path, target_aspect_ratio):
    """
    Optimize video for a target aspect ratio.
    
    Args:
        input_path (str): Path to the input video
        output_path (str): Path to save the output video
        target_aspect_ratio (str): Target aspect ratio (e.g. '16:9', '9:16', '1:1')
        
    Returns:
        dict: Result information
    """
    try:
        # Parse the target aspect ratio
        if target_aspect_ratio == '16:9':
            width, height = 1920, 1080
        elif target_aspect_ratio == '9:16':
            width, height = 1080, 1920
        elif target_aspect_ratio == '1:1':
            width, height = 1080, 1080
        else:
            # Parse custom ratio
            parts = target_aspect_ratio.split(':')
            if len(parts) == 2:
                ratio = float(parts[0]) / float(parts[1])
                # Base on 1080p height
                height = 1080
                width = int(height * ratio)
            else:
                raise ValueError(f"Invalid aspect ratio format: {target_aspect_ratio}")
        
        # Get video info using FFprobe
        ffprobe_path = current_app.config['FFPROBE_PATH']
        probe_command = [
            ffprobe_path,
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            input_path
        ]
        
        result = subprocess.run(probe_command, capture_output=True, text=True, check=True)
        video_info = json.loads(result.stdout)
        
        # Find video stream
        video_stream = None
        for stream in video_info.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break
        
        if not video_stream:
            raise ValueError("No video stream found")
        
        # Get original dimensions
        original_width = int(video_stream.get('width', 0))
        original_height = int(video_stream.get('height', 0))
        
        if original_width == 0 or original_height == 0:
            raise ValueError("Could not determine video dimensions")
        
        # Calculate scaling and padding
        original_ratio = original_width / original_height
        target_ratio = width / height
        
        # Determine filter based on aspect ratio comparison
        if abs(original_ratio - target_ratio) < 0.01:
            # Ratios are already very close
            scale_filter = f"scale={width}:{height}"
            video_filter = scale_filter
        elif original_ratio > target_ratio:
            # Original is wider, need to crop width or pad height
            new_width = int(original_height * target_ratio)
            crop_filter = f"crop={new_width}:{original_height}:((iw-{new_width})/2):0"
            scale_filter = f"scale={width}:{height}"
            video_filter = f"{crop_filter},{scale_filter}"
        else:
            # Original is taller, need to crop height or pad width
            new_height = int(original_width / target_ratio)
            crop_filter = f"crop={original_width}:{new_height}:0:((ih-{new_height})/2)"
            scale_filter = f"scale={width}:{height}"
            video_filter = f"{crop_filter},{scale_filter}"
        
        # Run FFmpeg to optimize the video
        ffmpeg_path = current_app.config['FFMPEG_PATH']
        command = [
            ffmpeg_path,
            '-i', input_path,
            '-vf', video_filter,
            '-c:v', 'libx264',
            '-c:a', 'copy',
            output_path
        ]
        
        subprocess.run(command, check=True, capture_output=True)
        
        return {
            "original_dimensions": f"{original_width}x{original_height}",
            "target_dimensions": f"{width}x{height}",
            "target_aspect_ratio": target_aspect_ratio
        }
    
    except Exception as e:
        # Log the error and re-raise
        print(f"Aspect ratio optimization error: {str(e)}")
        raise 