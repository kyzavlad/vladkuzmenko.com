import os
import tempfile
import subprocess
from flask import current_app
import json

def format_time(seconds):
    """
    Format time in seconds to SRT format (HH:MM:SS,mmm).
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

def create_srt_file(transcription, output_path=None):
    """
    Create an SRT subtitle file from transcription data.
    
    Args:
        transcription (dict): Transcription data with segments
        output_path (str, optional): Path to save the SRT file
        
    Returns:
        str: Path to the SRT file
    """
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix='.srt')
        os.close(fd)
    
    segments = transcription.get("segments", [])
    
    with open(output_path, 'w', encoding='utf-8') as srt_file:
        for i, segment in enumerate(segments):
            start_time = format_time(segment['start'])
            end_time = format_time(segment['end'])
            text = segment['text'].strip()
            
            # Write the subtitle entry
            srt_file.write(f"{i+1}\n")
            srt_file.write(f"{start_time} --> {end_time}\n")
            srt_file.write(f"{text}\n\n")
    
    return output_path

def create_ass_file(transcription, style_options, output_path=None):
    """
    Create an ASS subtitle file from transcription data with custom styles.
    
    Args:
        transcription (dict): Transcription data with segments
        style_options (dict): Custom style options
        output_path (str, optional): Path to save the ASS file
        
    Returns:
        str: Path to the ASS file
    """
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix='.ass')
        os.close(fd)
    
    # Set default style options
    default_style = {
        "font": "Arial",
        "font_size": 24,
        "primary_color": "&H00FFFFFF",  # white in ASS format
        "secondary_color": "&H00000000",  # black in ASS format
        "outline_color": "&H00000000",  # black in ASS format
        "back_color": "&H80000000",  # semi-transparent black in ASS format
        "bold": 0,
        "italic": 0,
        "underline": 0,
        "strike_out": 0,
        "scale_x": 100,
        "scale_y": 100,
        "spacing": 0,
        "angle": 0,
        "border_style": 1,
        "outline": 2,
        "shadow": 2,
        "alignment": 2,  # bottom-center
        "margin_l": 10,
        "margin_r": 10,
        "margin_v": 10
    }
    
    # Update with user options
    style = {**default_style, **style_options}
    
    segments = transcription.get("segments", [])
    
    with open(output_path, 'w', encoding='utf-8') as ass_file:
        # Write header
        ass_file.write("[Script Info]\n")
        ass_file.write("Title: Subtitles\n")
        ass_file.write("ScriptType: v4.00+\n")
        ass_file.write("WrapStyle: 0\n")
        ass_file.write("ScaledBorderAndShadow: yes\n")
        ass_file.write("PlayResX: 1920\n")
        ass_file.write("PlayResY: 1080\n\n")
        
        # Write styles
        ass_file.write("[V4+ Styles]\n")
        ass_file.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        ass_file.write(f"Style: Default,{style['font']},{style['font_size']},{style['primary_color']},{style['secondary_color']},{style['outline_color']},{style['back_color']},{style['bold']},{style['italic']},{style['underline']},{style['strike_out']},{style['scale_x']},{style['scale_y']},{style['spacing']},{style['angle']},{style['border_style']},{style['outline']},{style['shadow']},{style['alignment']},{style['margin_l']},{style['margin_r']},{style['margin_v']},1\n\n")
        
        # Write events
        ass_file.write("[Events]\n")
        ass_file.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        
        for segment in segments:
            start_time = convert_to_ass_time(segment['start'])
            end_time = convert_to_ass_time(segment['end'])
            text = segment['text'].strip()
            
            # Write the subtitle entry
            ass_file.write(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}\n")
    
    return output_path

def convert_to_ass_time(seconds):
    """
    Convert seconds to ASS time format (H:MM:SS.cc).
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted time string for ASS
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_part = seconds % 60
    centiseconds = int((seconds_part - int(seconds_part)) * 100)
    
    return f"{hours}:{minutes:02d}:{int(seconds_part):02d}.{centiseconds:02d}"

def burn_subtitles_to_video(video_path, subtitle_path, output_path, style_options=None):
    """
    Burn subtitles into video using FFmpeg.
    
    Args:
        video_path (str): Path to the input video
        subtitle_path (str): Path to the subtitle file
        output_path (str): Path to save the output video
        style_options (dict, optional): Additional style options
        
    Returns:
        bool: True if successful
    """
    ffmpeg_path = current_app.config['FFMPEG_PATH']
    
    subtitle_extension = os.path.splitext(subtitle_path)[1].lower()
    
    # Base command
    command = [ffmpeg_path, '-i', video_path]
    
    # Add subtitle input and filter
    if subtitle_extension == '.srt':
        # For SRT files, we can apply some styling
        subtitle_filter = f"subtitles={subtitle_path}"
        
        # Add style options if provided
        if style_options:
            style_args = []
            if 'font' in style_options:
                style_args.append(f"force_style='FontName={style_options['font']}'")
            if 'font_size' in style_options:
                style_args.append(f"force_style='FontSize={style_options['font_size']}'")
            if 'primary_color' in style_options and style_options['primary_color'].startswith('#'):
                # Convert hex color to ASS format
                color = style_options['primary_color'].lstrip('#')
                style_args.append(f"force_style='PrimaryColour=&H{color}'")
            if style_args:
                subtitle_filter += ":" + ":".join(style_args)
        
        command.extend(['-vf', subtitle_filter])
    elif subtitle_extension == '.ass':
        # ASS files already have the styling information
        command.extend(['-vf', f"ass={subtitle_path}"])
    else:
        # Unsupported subtitle format
        raise ValueError(f"Unsupported subtitle format: {subtitle_extension}")
    
    # Add output options and path
    command.extend([
        '-c:v', 'libx264',
        '-c:a', 'copy',
        '-pix_fmt', 'yuv420p',
        output_path
    ])
    
    # Run the command
    subprocess.run(command, check=True, capture_output=True)
    return True

def generate_subtitles(video_path, output_path, transcription, style_options=None):
    """
    Generate subtitles for a video.
    
    Args:
        video_path (str): Path to the input video
        output_path (str): Path to save the output video
        transcription (dict): Transcription data
        style_options (dict, optional): Custom style options
        
    Returns:
        dict: Result information
    """
    try:
        # Use style options or empty dict
        style_options = style_options or {}
        
        # Determine subtitle format based on style complexity
        if style_options and style_options.get('advanced', False):
            # For advanced styling, use ASS format
            subtitle_path = create_ass_file(transcription, style_options)
        else:
            # For basic styling, use SRT format
            subtitle_path = create_srt_file(transcription)
        
        # Burn subtitles into video
        burn_subtitles_to_video(video_path, subtitle_path, output_path, style_options)
        
        # Return result
        subtitle_extension = os.path.splitext(subtitle_path)[1]
        subtitle_filename = os.path.basename(video_path).split('.')[0] + subtitle_extension
        
        return {
            "subtitle_file": subtitle_filename,
            "subtitle_path": subtitle_path,
            "style_applied": style_options != {}
        }
    
    except Exception as e:
        # Log the error and re-raise
        print(f"Subtitle generation error: {str(e)}")
        raise 