import logging
import asyncio
import os
import json
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import datetime

from .subtitle_generator import SubtitleFormat, SubtitleStyle, SubtitleGenerator, ReadingSpeedPreset, TextPosition
from .subtitle_renderer import RenderQuality, SubtitleRenderer
from .subtitle_editor import SubtitleEditor
from .subtitle_positioning import SubtitlePositioningService
from .subtitle_positioning_lite import SubtitlePositioningLite
from .reading_speed import ReadingSpeedCalculator, AudienceType
from .language_support import LanguageSupport, TextDirection, LanguageScript


class SubtitleService:
    """
    Main service for subtitle-related operations, integrating all subtitle components.
    
    Features:
    - Generate subtitles from transcripts
    - Render videos with burned-in subtitles
    - Edit subtitle timing and text
    - Preview subtitles at specific timestamps
    - Convert between subtitle formats
    - Analyze subtitle quality
    - Smart positioning to avoid important visual content
    - Reading speed calibration for optimal subtitle durations
    - Multi-language support with proper character rendering
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the subtitle service.
        
        Args:
            config: Configuration options for subtitle generation, rendering, and editing
        """
        self.config = config or {}
        
        # Initialize components
        self.generator = SubtitleGenerator(
            default_format=self.config.get("default_format", SubtitleFormat.SRT),
            default_style=self.config.get("default_style", "default"),
            config=self.config.get("generator_config")
        )
        
        self.renderer = SubtitleRenderer(
            subtitle_generator=self.generator,
            default_quality=self.config.get("default_quality", RenderQuality.MEDIUM),
            ffmpeg_path=self.config.get("ffmpeg_path", "ffmpeg"),
            ffprobe_path=self.config.get("ffprobe_path", "ffprobe"),
            config=self.config.get("renderer_config")
        )
        
        self.editor = SubtitleEditor(
            subtitle_generator=self.generator,
            config=self.config.get("editor_config")
        )
        
        # Initialize positioning service - standard or lite version
        use_lite_positioning = self.config.get("use_lite_positioning", False)
        if use_lite_positioning:
            self.positioning = SubtitlePositioningLite(
                ffmpeg_path=self.config.get("ffmpeg_path", "ffmpeg"),
                ffprobe_path=self.config.get("ffprobe_path", "ffprobe"),
                config=self.config.get("positioning_config")
            )
            self.logger.info("Using lightweight subtitle positioning service")
        else:
            self.positioning = SubtitlePositioningService(
                ffmpeg_path=self.config.get("ffmpeg_path", "ffmpeg"),
                ffprobe_path=self.config.get("ffprobe_path", "ffprobe"),
                config=self.config.get("positioning_config")
            )
        
        # Initialize the reading speed calculator
        self.reading_speed_calculator = ReadingSpeedCalculator(
            config=self.config.get("reading_speed_config")
        )
        
        # Initialize language support
        self.language_support = LanguageSupport(
            config=self.config.get("language_config")
        )
        
        # Cache for active edit sessions
        self.active_sessions = {}
        
        self.logger = logging.getLogger(__name__)
    
    async def generate_subtitles(
        self,
        transcript: Dict[str, Any],
        output_path: str,
        format: Optional[SubtitleFormat] = None,
        style_name: Optional[str] = None,
        custom_style: Optional[Dict[str, Any]] = None,
        reading_speed_preset: Optional[str] = None,
        adjust_timing: Optional[bool] = None,
        detect_emphasis: Optional[bool] = None,
        language: Optional[str] = None,
        auto_detect_language: Optional[bool] = None
    ) -> str:
        """
        Generate subtitle file from transcript.
        
        Args:
            transcript: Transcript with timing information
            output_path: Path to save the subtitle file
            format: Subtitle format to use
            style_name: Name of style template to use
            custom_style: Custom style overrides as dictionary
            reading_speed_preset: Reading speed preset for timing adjustment ('slow', 'standard', 'fast', 'very_fast')
            adjust_timing: Whether to adjust subtitle timing based on reading speed
            detect_emphasis: Whether to automatically detect and apply formatting to emphasized text
            language: Language code (ISO 639-1) for the subtitle
            auto_detect_language: Whether to auto-detect language from text content
            
        Returns:
            Path to generated subtitle file
        """
        # Convert format string to enum if provided
        format_enum = None
        if format:
            try:
                format_enum = SubtitleFormat(format.lower())
            except ValueError:
                self.logger.warning(f"Invalid format value: {format}, using default")
        
        # Convert reading_speed_preset string to enum if provided
        reading_speed_enum = None
        if reading_speed_preset:
            try:
                reading_speed_enum = ReadingSpeedPreset(reading_speed_preset.lower())
            except ValueError:
                self.logger.warning(f"Invalid reading speed preset: {reading_speed_preset}, using default")
        
        # Convert custom style dict to SubtitleStyle if provided
        style_obj = None
        if custom_style:
            # Extract basic properties
            style_dict = {
                "name": custom_style.get("name", "Custom"),
                "font_family": custom_style.get("font_family", "Arial"),
                "font_size": custom_style.get("font_size", 24),
                "font_color": custom_style.get("font_color", "#FFFFFF"),
                "background_color": custom_style.get("background_color", "#00000080"),
                "bold": custom_style.get("bold", False),
                "italic": custom_style.get("italic", False),
                "emphasis_bold": custom_style.get("emphasis_bold", False),
                "emphasis_italic": custom_style.get("emphasis_italic", False),
                "emphasis_color": custom_style.get("emphasis_color", "#FFFFFF")
            }
            
            # Add optional properties if present
            for prop in ["alignment", "position", "outline_width", "outline_color", 
                         "shadow_offset", "shadow_color", "line_spacing", 
                         "max_lines", "max_chars_per_line"]:
                if prop in custom_style:
                    style_dict[prop] = custom_style[prop]
            
            # Create style object
            style_obj = SubtitleStyle(**style_dict)
        
        try:
            # Generate subtitles using the generator component
            subtitle_path = await self.generator.generate_subtitles(
                transcript=transcript,
                output_path=output_path,
                format=format_enum,
                style_name=style_name,
                custom_style=style_obj,
                reading_speed_preset=reading_speed_enum,
                adjust_timing=adjust_timing,
                detect_emphasis=detect_emphasis,
                language=language,
                auto_detect_language=auto_detect_language
            )
            
            self.logger.info(f"Generated subtitles at {subtitle_path}")
            return subtitle_path
            
        except Exception as e:
            self.logger.error(f"Error generating subtitles: {str(e)}")
            raise
    
    async def render_video_with_subtitles(
        self,
        video_path: str,
        transcript: Dict[str, Any],
        output_path: str,
        style_name: Optional[str] = None,
        custom_style: Optional[Dict[str, Any]] = None,
        quality: Optional[str] = None,
        background_blur: float = 0.0,
        force_text_color: bool = False,
        show_progress_callback: Optional[callable] = None
    ) -> str:
        """
        Render video with burned-in subtitles.
        
        Args:
            video_path: Path to input video file
            transcript: Transcript with timing information
            output_path: Path to save the output video
            style_name: Name of subtitle style to use
            custom_style: Custom style overrides as dictionary
            quality: Render quality setting ('low', 'medium', 'high', 'original')
            background_blur: Amount of background blur to apply (0.0-1.0)
            force_text_color: Whether to force text color even with background
            show_progress_callback: Callback function to report progress
            
        Returns:
            Path to the rendered video file
        """
        # Convert quality string to enum if provided
        quality_enum = None
        if quality:
            try:
                quality_enum = RenderQuality(quality.lower())
            except ValueError:
                self.logger.warning(f"Invalid quality value: {quality}, using default")
        
        # Convert custom style dict to SubtitleStyle if provided
        style_obj = None
        if custom_style:
            style_dict = {
                "name": custom_style.get("name", "Custom"),
                "font_family": custom_style.get("font_family", "Arial"),
                "font_size": custom_style.get("font_size", 24),
                "font_color": custom_style.get("font_color", "#FFFFFF"),
                "background_color": custom_style.get("background_color", "#00000080"),
                "bold": custom_style.get("bold", False),
                "italic": custom_style.get("italic", False)
            }
            
            # Add optional properties if present
            for prop in ["alignment", "position", "outline_width", "outline_color", 
                         "shadow_offset", "shadow_color", "line_spacing", 
                         "max_lines", "max_chars_per_line"]:
                if prop in custom_style:
                    style_dict[prop] = custom_style[prop]
            
            # Create style object
            style_obj = SubtitleStyle(**style_dict)
        
        try:
            # Render video using the renderer component
            rendered_path = await self.renderer.render_video_with_subtitles(
                video_path=video_path,
                transcript=transcript,
                output_path=output_path,
                style_name=style_name,
                custom_style=style_obj,
                quality=quality_enum,
                background_blur=background_blur,
                force_text_color=force_text_color,
                show_progress_callback=show_progress_callback
            )
            
            self.logger.info(f"Rendered video with subtitles at {rendered_path}")
            return rendered_path
            
        except Exception as e:
            self.logger.error(f"Error rendering video with subtitles: {str(e)}")
            raise
    
    async def generate_preview_image(
        self,
        video_path: str,
        transcript: Dict[str, Any],
        output_path: str,
        time_position: float = 0.0,
        style_name: Optional[str] = None,
        custom_style: Optional[Dict[str, Any]] = None,
        width: int = 1280
    ) -> str:
        """
        Generate a preview image from the video with subtitle overlay.
        
        Args:
            video_path: Path to input video file
            transcript: Transcript with timing information
            output_path: Path to save the preview image
            time_position: Time position in the video to capture (seconds)
            style_name: Name of subtitle style to use
            custom_style: Custom style overrides as dictionary
            width: Width of the output image
            
        Returns:
            Path to the generated preview image
        """
        # Convert custom style dict to SubtitleStyle if provided
        style_obj = None
        if custom_style:
            style_dict = {
                "name": custom_style.get("name", "Custom"),
                "font_family": custom_style.get("font_family", "Arial"),
                "font_size": custom_style.get("font_size", 24),
                "font_color": custom_style.get("font_color", "#FFFFFF"),
                "background_color": custom_style.get("background_color", "#00000080"),
                "bold": custom_style.get("bold", False),
                "italic": custom_style.get("italic", False)
            }
            
            # Add optional properties if present
            for prop in ["alignment", "position", "outline_width", "outline_color", 
                         "shadow_offset", "shadow_color", "line_spacing", 
                         "max_lines", "max_chars_per_line"]:
                if prop in custom_style:
                    style_dict[prop] = custom_style[prop]
            
            # Create style object
            style_obj = SubtitleStyle(**style_dict)
        
        try:
            # Generate preview image using the renderer component
            preview_path = await self.renderer.generate_preview_image(
                video_path=video_path,
                transcript=transcript,
                output_path=output_path,
                time_position=time_position,
                style_name=style_name,
                custom_style=style_obj,
                width=width
            )
            
            self.logger.info(f"Generated preview image at {preview_path}")
            return preview_path
            
        except Exception as e:
            self.logger.error(f"Error generating preview image: {str(e)}")
            raise
    
    def create_edit_session(
        self,
        transcript: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> str:
        """
        Create a new subtitle editing session.
        
        Args:
            transcript: Transcript with timing information
            session_id: Optional session ID (generated if not provided)
            
        Returns:
            Session ID for the editing session
        """
        # Generate a session ID if not provided
        if not session_id:
            import uuid
            session_id = str(uuid.uuid4())
        
        # Store a copy of the transcript in the session
        self.active_sessions[session_id] = {
            "transcript": transcript.copy(),
            "created_at": asyncio.get_event_loop().time()
        }
        
        self.logger.info(f"Created subtitle edit session: {session_id}")
        
        return session_id
    
    def get_session_transcript(self, session_id: str) -> Dict[str, Any]:
        """
        Get the current transcript for an editing session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Current transcript state
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Edit session not found: {session_id}")
        
        return self.active_sessions[session_id]["transcript"].copy()
    
    def update_session_transcript(
        self,
        session_id: str,
        transcript: Dict[str, Any]
    ) -> None:
        """
        Update the transcript for an editing session.
        
        Args:
            session_id: Session ID
            transcript: Updated transcript
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Edit session not found: {session_id}")
        
        self.active_sessions[session_id]["transcript"] = transcript.copy()
        self.logger.debug(f"Updated transcript for session: {session_id}")
    
    def close_edit_session(self, session_id: str) -> None:
        """
        Close an editing session and clean up resources.
        
        Args:
            session_id: Session ID
        """
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            self.logger.info(f"Closed subtitle edit session: {session_id}")
    
    def cleanup_old_sessions(self, max_age_seconds: int = 3600) -> int:
        """
        Clean up old editing sessions.
        
        Args:
            max_age_seconds: Maximum age of sessions in seconds
            
        Returns:
            Number of sessions removed
        """
        current_time = asyncio.get_event_loop().time()
        sessions_to_remove = []
        
        for session_id, session_data in self.active_sessions.items():
            age = current_time - session_data.get("created_at", 0)
            if age > max_age_seconds:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.active_sessions[session_id]
        
        if sessions_to_remove:
            self.logger.info(f"Cleaned up {len(sessions_to_remove)} old edit sessions")
            
        return len(sessions_to_remove)
    
    # Editor operations wrapped for convenience
    
    def edit_subtitle_text(
        self,
        session_id: str,
        segment_id: int,
        new_text: str
    ) -> Dict[str, Any]:
        """
        Edit the text of a subtitle segment in a session.
        
        Args:
            session_id: Session ID
            segment_id: Index of the segment to edit
            new_text: New text for the segment
            
        Returns:
            Updated transcript
        """
        transcript = self.get_session_transcript(session_id)
        
        updated_transcript = self.editor.edit_subtitle_text(
            transcript=transcript,
            segment_id=segment_id,
            new_text=new_text
        )
        
        self.update_session_transcript(session_id, updated_transcript)
        return updated_transcript
    
    def adjust_timing(
        self,
        session_id: str,
        segment_id: int,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Adjust the timing of a subtitle segment in a session.
        
        Args:
            session_id: Session ID
            segment_id: Index of the segment to adjust
            start_time: New start time in seconds (if None, keep original)
            end_time: New end time in seconds (if None, keep original)
            
        Returns:
            Updated transcript
        """
        transcript = self.get_session_transcript(session_id)
        
        updated_transcript = self.editor.adjust_timing(
            transcript=transcript,
            segment_id=segment_id,
            start_time=start_time,
            end_time=end_time
        )
        
        self.update_session_transcript(session_id, updated_transcript)
        return updated_transcript
    
    def split_segment(
        self,
        session_id: str,
        segment_id: int,
        split_time: float
    ) -> Dict[str, Any]:
        """
        Split a subtitle segment into two at the specified time.
        
        Args:
            session_id: Session ID
            segment_id: Index of the segment to split
            split_time: Time position to split at (seconds)
            
        Returns:
            Updated transcript
        """
        transcript = self.get_session_transcript(session_id)
        
        updated_transcript = self.editor.split_segment(
            transcript=transcript,
            segment_id=segment_id,
            split_time=split_time
        )
        
        self.update_session_transcript(session_id, updated_transcript)
        return updated_transcript
    
    def merge_segments(
        self,
        session_id: str,
        segment_ids: List[int]
    ) -> Dict[str, Any]:
        """
        Merge multiple subtitle segments into one.
        
        Args:
            session_id: Session ID
            segment_ids: List of segment indices to merge
            
        Returns:
            Updated transcript
        """
        transcript = self.get_session_transcript(session_id)
        
        updated_transcript = self.editor.merge_segments(
            transcript=transcript,
            segment_ids=segment_ids
        )
        
        self.update_session_transcript(session_id, updated_transcript)
        return updated_transcript
    
    def add_segment(
        self,
        session_id: str,
        start_time: float,
        end_time: float,
        text: str,
        position: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Add a new subtitle segment.
        
        Args:
            session_id: Session ID
            start_time: Start time in seconds
            end_time: End time in seconds
            text: Subtitle text
            position: Index to insert at (None = auto-position based on time)
            
        Returns:
            Updated transcript
        """
        transcript = self.get_session_transcript(session_id)
        
        updated_transcript = self.editor.add_segment(
            transcript=transcript,
            start_time=start_time,
            end_time=end_time,
            text=text,
            position=position
        )
        
        self.update_session_transcript(session_id, updated_transcript)
        return updated_transcript
    
    def delete_segment(
        self,
        session_id: str,
        segment_id: int
    ) -> Dict[str, Any]:
        """
        Delete a subtitle segment.
        
        Args:
            session_id: Session ID
            segment_id: Index of the segment to delete
            
        Returns:
            Updated transcript
        """
        transcript = self.get_session_transcript(session_id)
        
        updated_transcript = self.editor.delete_segment(
            transcript=transcript,
            segment_id=segment_id
        )
        
        self.update_session_transcript(session_id, updated_transcript)
        return updated_transcript
    
    def shift_timings(
        self,
        session_id: str,
        time_shift: float,
        segment_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Shift subtitle timings by a specified amount.
        
        Args:
            session_id: Session ID
            time_shift: Amount to shift (seconds, positive or negative)
            segment_ids: List of segment indices to shift (None = all segments)
            
        Returns:
            Updated transcript
        """
        transcript = self.get_session_transcript(session_id)
        
        updated_transcript = self.editor.shift_timings(
            transcript=transcript,
            time_shift=time_shift,
            segment_ids=segment_ids
        )
        
        self.update_session_transcript(session_id, updated_transcript)
        return updated_transcript
    
    def undo(self, session_id: str) -> Dict[str, Any]:
        """
        Undo the last operation in a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Updated transcript
        """
        transcript = self.get_session_transcript(session_id)
        
        updated_transcript = self.editor.undo(transcript)
        
        self.update_session_transcript(session_id, updated_transcript)
        return updated_transcript
    
    def redo(self, session_id: str) -> Dict[str, Any]:
        """
        Redo the last undone operation in a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Updated transcript
        """
        transcript = self.get_session_transcript(session_id)
        
        updated_transcript = self.editor.redo(transcript)
        
        self.update_session_transcript(session_id, updated_transcript)
        return updated_transcript
    
    def analyze_subtitle_quality(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Analyze the quality of subtitles in a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Dictionary with quality metrics and issues
        """
        transcript = self.get_session_transcript(session_id)
        
        return self.editor.analyze_subtitle_quality(transcript)
    
    def auto_adjust_timings(self, session_id: str) -> Dict[str, Any]:
        """
        Automatically adjust subtitle timings to improve readability.
        
        Args:
            session_id: Session ID
            
        Returns:
            Updated transcript with adjusted timings
        """
        transcript = self.get_session_transcript(session_id)
        
        updated_transcript = self.editor.auto_adjust_timings(transcript)
        
        self.update_session_transcript(session_id, updated_transcript)
        return updated_transcript
    
    def get_available_styles(self) -> List[str]:
        """
        Get names of all available style templates.
        
        Returns:
            List of style template names
        """
        return self.generator.get_available_styles()
    
    async def save_subtitle_file(
        self,
        session_id: str,
        output_path: str,
        format: Optional[str] = None,
        style_name: Optional[str] = None,
        custom_style: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save the current session transcript to a subtitle file.
        
        Args:
            session_id: Session ID
            output_path: Path to save the subtitle file
            format: Subtitle format to use
            style_name: Name of style template to use
            custom_style: Custom style overrides as dictionary
            
        Returns:
            Path to the subtitle file
        """
        transcript = self.get_session_transcript(session_id)
        
        # Convert format string to enum if provided
        format_enum = None
        if format:
            try:
                format_enum = SubtitleFormat(format.lower())
            except ValueError:
                self.logger.warning(f"Invalid format value: {format}, using default")
        
        # Convert custom style dict to SubtitleStyle if provided
        style_obj = None
        if custom_style:
            style_dict = {
                "name": custom_style.get("name", "Custom"),
                "font_family": custom_style.get("font_family", "Arial"),
                "font_size": custom_style.get("font_size", 24),
                "font_color": custom_style.get("font_color", "#FFFFFF"),
                "background_color": custom_style.get("background_color", "#00000080"),
                "bold": custom_style.get("bold", False),
                "italic": custom_style.get("italic", False)
            }
            
            # Add optional properties if present
            for prop in ["alignment", "position", "outline_width", "outline_color", 
                         "shadow_offset", "shadow_color", "line_spacing", 
                         "max_lines", "max_chars_per_line"]:
                if prop in custom_style:
                    style_dict[prop] = custom_style[prop]
            
            # Create style object
            style_obj = SubtitleStyle(**style_dict)
        
        return await self.editor.save_to_file(
            transcript=transcript,
            output_path=output_path,
            format=format_enum,
            style_name=style_name,
            custom_style=style_obj
        )
    
    def adjust_subtitle_timing(
        self,
        session_id: str,
        reading_speed_preset: str = "standard"
    ) -> Dict[str, Any]:
        """
        Adjust subtitle timing based on reading speed.
        
        Args:
            session_id: Session ID
            reading_speed_preset: Reading speed preset ('slow', 'standard', 'fast', 'very_fast')
            
        Returns:
            Updated transcript with adjusted timing
        """
        transcript = self.get_session_transcript(session_id)
        
        # Convert reading_speed_preset string to enum
        try:
            reading_speed_enum = ReadingSpeedPreset(reading_speed_preset.lower())
        except ValueError:
            self.logger.warning(f"Invalid reading speed preset: {reading_speed_preset}, using standard")
            reading_speed_enum = ReadingSpeedPreset.STANDARD
        
        # Adjust timing using the generator component
        updated_transcript = self.generator.adjust_transcript_timing(
            transcript=transcript,
            reading_speed_preset=reading_speed_enum
        )
        
        # Update the session with the new transcript
        self.update_session_transcript(session_id, updated_transcript)
        return updated_transcript
    
    def calibrate_subtitle_durations(
        self,
        session_id: str,
        audience_type: str = "general",
        calculation_method: str = "character",
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calibrate subtitle durations based on text content and audience reading speed.
        
        This uses the more advanced ReadingSpeedCalculator for fine-grained duration
        calibration based on text complexity and content.
        
        Args:
            session_id: Session ID
            audience_type: Target audience type ('children', 'general', 'experienced', 'speed_reader')
            calculation_method: Method for calculating duration ('character', 'word', 'syllable')
            language: Language code (e.g. 'en', 'fr', 'de')
            
        Returns:
            Updated transcript with calibrated durations
        """
        transcript = self.get_session_transcript(session_id)
        
        # Map audience_type string to enum
        try:
            audience_enum = AudienceType(audience_type.lower())
        except ValueError:
            self.logger.warning(f"Invalid audience type: {audience_type}, using general")
            audience_enum = AudienceType.GENERAL
        
        # Update calculator settings
        self.reading_speed_calculator.set_audience_type(audience_enum)
        
        # Prepare subtitles in the format expected by the calculator
        subtitles = []
        for segment in transcript.get("segments", []):
            subtitles.append({
                "text": segment.get("text", ""),
                "start": segment.get("start", 0),
                "end": segment.get("end", 0)
            })
        
        # Calibrate durations
        calibrated_subtitles = self.reading_speed_calculator.calibrate_subtitle_durations(subtitles)
        
        # Update the transcript with calibrated durations
        for i, segment in enumerate(transcript.get("segments", [])):
            if i < len(calibrated_subtitles):
                segment["end"] = calibrated_subtitles[i]["end"]
        
        # Update the session with the new transcript
        self.update_session_transcript(session_id, transcript)
        
        return transcript
    
    def get_reading_speed_presets(self) -> Dict[str, int]:
        """
        Get available reading speed presets with their WPM values.
        
        Returns:
            Dictionary of reading speed presets and their WPM values
        """
        return {
            preset.value: self.generator.get_reading_speed(preset)
            for preset in ReadingSpeedPreset
        }
    
    def get_audience_reading_speeds(self) -> Dict[str, Dict[str, int]]:
        """
        Get available audience types with their reading speed metrics.
        
        Returns:
            Dictionary of audience types and their reading speed metrics
        """
        result = {}
        
        for audience_type in AudienceType:
            # Temporarily set audience type to get values
            original_audience = self.reading_speed_calculator.audience_type
            self.reading_speed_calculator.set_audience_type(audience_type)
            
            # Get speeds for this audience type
            speeds = self.reading_speed_calculator.get_reading_speeds()
            result[audience_type.value] = speeds
            
            # Restore original audience type
            self.reading_speed_calculator.set_audience_type(original_audience)
        
        return result
    
    def set_reading_speed(self, preset: str, words_per_minute: int) -> None:
        """
        Set custom reading speed for a preset.
        
        Args:
            preset: Reading speed preset name
            words_per_minute: Words per minute value
        """
        try:
            preset_enum = ReadingSpeedPreset(preset.lower())
            self.generator.set_reading_speed(preset_enum, words_per_minute)
        except ValueError:
            self.logger.warning(f"Invalid reading speed preset: {preset}")
    
    def set_audience_reading_speed(
        self,
        audience_type: str,
        words_per_minute: Optional[int] = None,
        chars_per_minute: Optional[int] = None,
        syllables_per_minute: Optional[int] = None
    ) -> None:
        """
        Set custom reading speed metrics for an audience type.
        
        Args:
            audience_type: Audience type name
            words_per_minute: Words per minute value
            chars_per_minute: Characters per minute value
            syllables_per_minute: Syllables per minute value
        """
        try:
            audience_enum = AudienceType(audience_type.lower())
            
            # Create a new config with custom values
            config = self.reading_speed_calculator.config.copy()
            
            if words_per_minute is not None:
                config['custom_wpm'] = words_per_minute
                
            if chars_per_minute is not None:
                config['custom_cpm'] = chars_per_minute
                
            if syllables_per_minute is not None:
                config['custom_spm'] = syllables_per_minute
            
            # Update the calculator with new settings
            self.reading_speed_calculator = ReadingSpeedCalculator(config=config)
            self.reading_speed_calculator.set_audience_type(audience_enum)
            
        except ValueError:
            self.logger.warning(f"Invalid audience type: {audience_type}")
    
    def detect_emphasis(
        self,
        session_id: str,
        style_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect and apply formatting to emphasized text in the transcript.
        
        Args:
            session_id: Session ID
            style_name: Name of style template to use for formatting decisions
            
        Returns:
            Updated transcript with emphasis formatting applied
        """
        transcript = self.get_session_transcript(session_id)
        
        # Process transcript using the generator component
        updated_transcript = self.generator.detect_emphasis(
            transcript=transcript,
            style=self._get_style_obj(style_name)
        )
        
        # Update the session with the new transcript
        self.update_session_transcript(session_id, updated_transcript)
        return updated_transcript
    
    def _get_style_obj(self, style_name: Optional[str] = None) -> SubtitleStyle:
        """
        Get a SubtitleStyle object for the given style name.
        
        Args:
            style_name: Name of style template to use
            
        Returns:
            SubtitleStyle object
        """
        if not style_name:
            return None
        
        # This will return None if the style doesn't exist,
        # and the generator will use the default style
        return self.generator._get_style(style_name, None)
    
    def set_emphasis_options(
        self,
        use_bold: bool = True,
        use_italic: bool = False,
        emphasis_color: Optional[str] = None,
        style_name: Optional[str] = "default"
    ) -> None:
        """
        Configure emphasis formatting options.
        
        Args:
            use_bold: Whether to use bold formatting for emphasis
            use_italic: Whether to use italic formatting for emphasis
            emphasis_color: Optional color for emphasized text (hex color code)
            style_name: Name of style template to modify
        """
        # Get the style to modify
        style = self._get_style_obj(style_name)
        if not style:
            self.logger.warning(f"Style '{style_name}' not found, creating a new custom style")
            style = self.generator._get_style("default", None)  # Get the default style
        
        # Update style properties
        style.emphasis_bold = use_bold
        style.emphasis_italic = use_italic
        style.emphasis_color = emphasis_color
        
        # Update the style in the generator
        self.generator.add_custom_style(style)
        
        self.logger.info(f"Updated emphasis options for style '{style_name}'")
    
    async def optimize_subtitle_positioning(
        self,
        video_path: str,
        transcript: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze video content to determine optimal subtitle positioning.
        
        Args:
            video_path: Path to the video file
            transcript: Transcript with timing information
            
        Returns:
            Updated transcript with optimal subtitle positioning information
        """
        self.logger.info(f"Optimizing subtitle positioning for video: {video_path}")
        
        try:
            # Use the positioning service to analyze the video and determine optimal positions
            optimized_transcript = await self.positioning.analyze_video_for_positioning(
                video_path=video_path,
                transcript=transcript
            )
            
            self.logger.info("Subtitle positioning optimization completed successfully")
            return optimized_transcript
            
        except Exception as e:
            self.logger.error(f"Error optimizing subtitle positioning: {str(e)}")
            self.logger.warning("Using default positioning since optimization failed")
            return transcript

    async def generate_subtitles_with_smart_positioning(
        self,
        video_path: str,
        transcript: Dict[str, Any],
        output_path: str,
        format: Optional[SubtitleFormat] = None,
        style_name: Optional[str] = None,
        custom_style: Optional[Dict[str, Any]] = None,
        reading_speed_preset: Optional[str] = None,
        adjust_timing: Optional[bool] = None,
        detect_emphasis: Optional[bool] = None
    ) -> str:
        """
        Generate subtitle file with smart positioning based on video content analysis.
        
        Args:
            video_path: Path to the video file
            transcript: Transcript with timing information
            output_path: Path to save the subtitle file
            format: Subtitle format to use
            style_name: Name of style template to use
            custom_style: Custom style overrides as dictionary
            reading_speed_preset: Reading speed preset for timing adjustment
            adjust_timing: Whether to adjust subtitle timing based on reading speed
            detect_emphasis: Whether to detect and format emphasized text
            
        Returns:
            Path to generated subtitle file
        """
        # Optimize subtitle positioning based on video content
        optimized_transcript = await self.optimize_subtitle_positioning(
            video_path=video_path,
            transcript=transcript
        )
        
        # Generate subtitles with the optimized positioning information
        return await self.generate_subtitles(
            transcript=optimized_transcript,
            output_path=output_path,
            format=format,
            style_name=style_name,
            custom_style=custom_style,
            reading_speed_preset=reading_speed_preset,
            adjust_timing=adjust_timing,
            detect_emphasis=detect_emphasis
        )
        
    async def render_video_with_smart_positioning(
        self,
        video_path: str,
        transcript: Dict[str, Any],
        output_path: str,
        style_name: Optional[str] = None,
        custom_style: Optional[Dict[str, Any]] = None,
        quality: Optional[Union[str, RenderQuality]] = None,
        background_blur: Optional[bool] = None
    ) -> str:
        """
        Render video with subtitles that are intelligently positioned to avoid important content.
        
        Args:
            video_path: Path to the video file
            transcript: Transcript with timing information
            output_path: Path to save the rendered video
            style_name: Name of style template to use
            custom_style: Custom style overrides as dictionary
            quality: Video render quality
            background_blur: Whether to apply background blur behind subtitles
            
        Returns:
            Path to rendered video
        """
        # Optimize subtitle positioning based on video content
        optimized_transcript = await self.optimize_subtitle_positioning(
            video_path=video_path,
            transcript=transcript
        )
        
        # Render video with the optimized positioning information
        return await self.renderer.render_video_with_subtitles(
            video_path=video_path,
            transcript=optimized_transcript,
            output_path=output_path,
            style_name=style_name,
            custom_style=custom_style,
            quality=quality,
            background_blur=background_blur
        )
    
    def get_positioning_options(self) -> Dict[str, Any]:
        """
        Get available subtitle positioning options and configurations.
        
        Returns:
            Dictionary of positioning options and their descriptions
        """
        return {
            "automatic": "Automatically determine optimal subtitle position based on video content",
            "bottom": "Always position subtitles at the bottom of the video",
            "top": "Always position subtitles at the top of the video",
            "center": "Always position subtitles in the center of the video",
            "position_preference": self.positioning.position_preference,
            "enable_face_detection": self.positioning.enable_face_detection,
            "enable_object_detection": self.positioning.enable_object_detection,
            "enable_text_detection": self.positioning.enable_text_detection
        }
    
    def update_positioning_config(self, config: Dict[str, Any]) -> None:
        """
        Update subtitle positioning configuration.
        
        Args:
            config: New configuration options
        """
        # Update position preference if provided
        if "position_preference" in config:
            self.positioning.position_preference = config["position_preference"]
        
        # Update detection enablement if provided
        if "enable_face_detection" in config:
            self.positioning.enable_face_detection = config["enable_face_detection"]
        
        if "enable_object_detection" in config:
            self.positioning.enable_object_detection = config["enable_object_detection"]
        
        if "enable_text_detection" in config:
            self.positioning.enable_text_detection = config["enable_text_detection"]
        
        self.logger.info("Updated subtitle positioning configuration")
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """
        Get a list of supported languages.
        
        Returns:
            List of dictionaries containing language information (code, name, script, direction)
        """
        languages = []
        
        for lang_code in self.language_support.LANGUAGE_SCRIPTS.keys():
            script = self.language_support.get_script_for_language(lang_code)
            direction = self.language_support.get_text_direction(lang_code)
            name = self.language_support.get_language_name(lang_code)
            
            languages.append({
                "code": lang_code,
                "name": name,
                "script": script.value,
                "direction": direction.value,
                "is_rtl": direction == TextDirection.RTL,
                "font": self.language_support.get_best_font(lang_code)
            })
        
        # Sort by language name
        return sorted(languages, key=lambda x: x["name"])
    
    def detect_transcript_language(self, transcript: Dict[str, Any]) -> str:
        """
        Detect the dominant language in a transcript.
        
        Args:
            transcript: Transcript to analyze
            
        Returns:
            Detected language code (ISO 639-1)
        """
        # Concatenate all text segments for better language detection
        all_text = ""
        for segment in transcript.get("segments", []):
            text = segment.get("text", "")
            if text:
                all_text += text + " "
        
        # Detect language
        if all_text:
            try:
                return self.language_support.detect_language(all_text)
            except Exception as e:
                self.logger.warning(f"Language detection failed: {str(e)}")
        
        # Fall back to default language
        return self.language_support.default_language
    
    def get_language_font_recommendations(self, language_code: str) -> List[str]:
        """
        Get recommended fonts for a specific language.
        
        Args:
            language_code: Language code (ISO 639-1)
            
        Returns:
            List of recommended font names
        """
        return self.language_support.get_recommended_fonts(language_code)
    
    def set_default_language(self, language_code: str) -> None:
        """
        Set the default language for subtitle generation.
        
        Args:
            language_code: Language code (ISO 639-1)
        """
        if not self.language_support.supports_language(language_code):
            self.logger.warning(f"Language {language_code} is not fully supported, some features may not work correctly")
        
        self.generator.default_language = language_code
        self.language_support.default_language = language_code 

    async def generate_multiple_outputs(
        self,
        video_path: str,
        transcript: Dict[str, Any],
        output_dir: str,
        base_filename: Optional[str] = None,
        subtitle_formats: List[str] = ["srt", "vtt"],
        generate_video: bool = True,
        video_quality: Optional[str] = None,
        optimize_positioning: bool = False,
        style_name: Optional[str] = None,
        custom_style: Optional[Dict[str, Any]] = None,
        reading_speed_preset: Optional[str] = None,
        detect_emphasis: bool = False,
        language: Optional[str] = None,
        auto_detect_language: bool = False,
        background_blur: float = 0.0,
        video_extension: str = "mp4",
        show_progress_callback: Optional[callable] = None
    ) -> Dict[str, str]:
        """
        Generate multiple outputs: video with burnt-in subtitles and separate subtitle files.
        
        This method combines generating subtitle files in multiple formats and 
        rendering video with burnt-in subtitles in one convenient operation.
        
        Args:
            video_path: Path to the input video file
            transcript: Transcript with timing information
            output_dir: Directory to save all output files
            base_filename: Base filename for all outputs (derived from video filename if None)
            subtitle_formats: List of subtitle formats to generate (e.g., ["srt", "vtt", "ass"])
            generate_video: Whether to generate video with burnt-in subtitles
            video_quality: Video render quality ('low', 'medium', 'high', 'original')
            optimize_positioning: Whether to optimize subtitle positioning based on video content
            style_name: Name of style template to use
            custom_style: Custom style overrides as dictionary
            reading_speed_preset: Reading speed preset for timing adjustment
            detect_emphasis: Whether to detect and format emphasized text
            language: Language code (ISO 639-1) for the subtitle
            auto_detect_language: Whether to auto-detect language from text content
            background_blur: Amount of background blur to apply behind subtitles (0.0-1.0)
            video_extension: Extension for the output video file (without the dot)
            show_progress_callback: Callback function for progress reporting
            
        Returns:
            Dictionary mapping output types to their file paths
        """
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine base filename if not provided
        if not base_filename:
            video_basename = os.path.basename(video_path)
            base_filename = os.path.splitext(video_basename)[0]
        
        # Create a deep copy of the transcript to avoid modifying the original
        import copy
        working_transcript = copy.deepcopy(transcript)
        
        # Optimize positioning if requested
        if optimize_positioning:
            self.logger.info(f"Optimizing subtitle positioning for {base_filename}")
            working_transcript = await self.optimize_subtitle_positioning(
                video_path=video_path,
                transcript=working_transcript
            )
            
        # Initialize result dictionary
        result_files = {}
        
        # Generate subtitle files in multiple formats
        for format_name in subtitle_formats:
            try:
                format_enum = SubtitleFormat(format_name.lower())
                subtitle_path = os.path.join(output_dir, f"{base_filename}.{format_name.lower()}")
                
                self.logger.info(f"Generating {format_name.upper()} subtitle file: {subtitle_path}")
                output_path = await self.generate_subtitles(
                    transcript=working_transcript,
                    output_path=subtitle_path,
                    format=format_enum,
                    style_name=style_name,
                    custom_style=custom_style,
                    reading_speed_preset=reading_speed_preset,
                    detect_emphasis=detect_emphasis,
                    language=language,
                    auto_detect_language=auto_detect_language
                )
                
                result_files[f"subtitle_{format_name.lower()}"] = output_path
                self.logger.info(f"Generated {format_name.upper()} subtitle file: {output_path}")
                
            except ValueError:
                self.logger.warning(f"Invalid subtitle format: {format_name}, skipping")
                continue
        
        # Generate video with burnt-in subtitles if requested
        if generate_video:
            video_output_path = os.path.join(output_dir, f"{base_filename}_subtitled.{video_extension}")
            
            try:
                self.logger.info(f"Rendering video with burnt-in subtitles: {video_output_path}")
                rendered_path = await self.render_video_with_subtitles(
                    video_path=video_path,
                    transcript=working_transcript,
                    output_path=video_output_path,
                    style_name=style_name,
                    custom_style=custom_style,
                    quality=video_quality,
                    background_blur=background_blur,
                    show_progress_callback=show_progress_callback
                )
                
                result_files["video"] = rendered_path
                self.logger.info(f"Generated video with burnt-in subtitles: {rendered_path}")
                
            except Exception as e:
                self.logger.error(f"Error rendering video with subtitles: {str(e)}")
                result_files["video_error"] = str(e)
        
        # Generate a manifest JSON file with information about the outputs
        manifest = {
            "source_video": video_path,
            "base_filename": base_filename,
            "generated_at": datetime.datetime.now().isoformat(),
            "output_files": result_files,
            "subtitle_options": {
                "style": style_name,
                "optimize_positioning": optimize_positioning,
                "reading_speed": reading_speed_preset,
                "detect_emphasis": detect_emphasis,
                "language": language,
                "auto_detect_language": auto_detect_language
            }
        }
        
        # Save manifest
        manifest_path = os.path.join(output_dir, f"{base_filename}_manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        result_files["manifest"] = manifest_path
        
        return result_files 