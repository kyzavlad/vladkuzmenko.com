import logging
import asyncio
import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from copy import deepcopy

from .subtitle_generator import SubtitleFormat, SubtitleStyle, SubtitleGenerator


@dataclass
class EditOperation:
    """Represents a subtitle edit operation."""
    operation_type: str  # "add", "delete", "modify", "shift", "merge", "split"
    segment_ids: List[int]
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    

class SubtitleEditor:
    """
    Provides functionality for editing and adjusting subtitle timings and text.
    
    Features:
    - Edit subtitle text
    - Adjust subtitle timing
    - Split and merge subtitle segments
    - Batch shifting of subtitle timings
    - Undo/redo functionality
    """
    
    def __init__(
        self,
        subtitle_generator: Optional[SubtitleGenerator] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the subtitle editor.
        
        Args:
            subtitle_generator: SubtitleGenerator instance for generating subtitle files
            config: Additional configuration options
        """
        self.subtitle_generator = subtitle_generator or SubtitleGenerator()
        self.config = config or {}
        
        # Operation history for undo/redo
        self.history = []
        self.current_position = -1
        
        # Maximum history size
        self.max_history = self.config.get("max_history", 100)
        
        self.logger = logging.getLogger(__name__)
    
    def edit_subtitle_text(
        self, 
        transcript: Dict[str, Any],
        segment_id: int,
        new_text: str
    ) -> Dict[str, Any]:
        """
        Edit the text of a subtitle segment.
        
        Args:
            transcript: Transcript with timing information
            segment_id: Index of the segment to edit
            new_text: New text for the segment
            
        Returns:
            Updated transcript
        """
        if not transcript or "segments" not in transcript:
            raise ValueError("Invalid transcript format")
        
        if segment_id < 0 or segment_id >= len(transcript["segments"]):
            raise ValueError(f"Invalid segment ID: {segment_id}")
        
        # Create a deep copy to avoid modifying the original
        transcript_copy = deepcopy(transcript)
        segments = transcript_copy["segments"]
        
        # Store the original text for history
        original_text = segments[segment_id].get("text", "")
        
        # Create an edit operation for history
        operation = EditOperation(
            operation_type="modify",
            segment_ids=[segment_id],
            params={"old_text": original_text, "new_text": new_text},
            description=f"Edit text of segment {segment_id}"
        )
        
        # Update the segment text
        segments[segment_id]["text"] = new_text
        
        # Add operation to history
        self._add_to_history(operation, transcript_copy)
        
        return transcript_copy
    
    def adjust_timing(
        self,
        transcript: Dict[str, Any],
        segment_id: int,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Adjust the timing of a subtitle segment.
        
        Args:
            transcript: Transcript with timing information
            segment_id: Index of the segment to adjust
            start_time: New start time in seconds (if None, keep original)
            end_time: New end time in seconds (if None, keep original)
            
        Returns:
            Updated transcript
        """
        if not transcript or "segments" not in transcript:
            raise ValueError("Invalid transcript format")
        
        if segment_id < 0 or segment_id >= len(transcript["segments"]):
            raise ValueError(f"Invalid segment ID: {segment_id}")
        
        # Create a deep copy to avoid modifying the original
        transcript_copy = deepcopy(transcript)
        segments = transcript_copy["segments"]
        
        # Store original timing for history
        original_start = segments[segment_id].get("start", 0)
        original_end = segments[segment_id].get("end", 0)
        
        # Validate new timing
        if start_time is not None and end_time is not None and start_time >= end_time:
            raise ValueError("Start time must be before end time")
        
        # Determine new timing values
        new_start = start_time if start_time is not None else original_start
        new_end = end_time if end_time is not None else original_end
        
        # Create an edit operation for history
        operation = EditOperation(
            operation_type="modify",
            segment_ids=[segment_id],
            params={
                "old_start": original_start,
                "old_end": original_end,
                "new_start": new_start,
                "new_end": new_end
            },
            description=f"Adjust timing of segment {segment_id}"
        )
        
        # Update the segment timing
        segments[segment_id]["start"] = new_start
        segments[segment_id]["end"] = new_end
        
        # Add operation to history
        self._add_to_history(operation, transcript_copy)
        
        return transcript_copy
    
    def split_segment(
        self,
        transcript: Dict[str, Any],
        segment_id: int,
        split_time: float
    ) -> Dict[str, Any]:
        """
        Split a subtitle segment into two at the specified time.
        
        Args:
            transcript: Transcript with timing information
            segment_id: Index of the segment to split
            split_time: Time position to split at (seconds)
            
        Returns:
            Updated transcript
        """
        if not transcript or "segments" not in transcript:
            raise ValueError("Invalid transcript format")
        
        if segment_id < 0 or segment_id >= len(transcript["segments"]):
            raise ValueError(f"Invalid segment ID: {segment_id}")
        
        # Create a deep copy to avoid modifying the original
        transcript_copy = deepcopy(transcript)
        segments = transcript_copy["segments"]
        
        # Get the segment to split
        segment = segments[segment_id]
        
        # Validate split time
        start_time = segment.get("start", 0)
        end_time = segment.get("end", 0)
        
        if split_time <= start_time or split_time >= end_time:
            raise ValueError(f"Split time must be between segment start ({start_time}) and end ({end_time})")
        
        # Split the text (simple split at nearest space)
        text = segment.get("text", "")
        
        # Calculate the split position based on time proportion
        split_ratio = (split_time - start_time) / (end_time - start_time)
        
        # Try to find a natural break point (space) near the ratio point
        words = text.split()
        if not words:
            raise ValueError("Cannot split empty text")
        
        split_word_index = max(1, min(len(words) - 1, int(len(words) * split_ratio)))
        
        # Create texts for two segments
        first_text = " ".join(words[:split_word_index])
        second_text = " ".join(words[split_word_index:])
        
        # Create a new segment
        new_segment = {
            "start": split_time,
            "end": end_time,
            "text": second_text
        }
        
        # Update the original segment
        segment["end"] = split_time
        segment["text"] = first_text
        
        # Insert the new segment after the original
        segments.insert(segment_id + 1, new_segment)
        
        # Create an edit operation for history
        operation = EditOperation(
            operation_type="split",
            segment_ids=[segment_id],
            params={
                "original_segment": deepcopy(segment),
                "split_time": split_time,
                "new_segment_id": segment_id + 1
            },
            description=f"Split segment {segment_id} at {split_time:.2f}s"
        )
        
        # Add operation to history
        self._add_to_history(operation, transcript_copy)
        
        return transcript_copy
    
    def merge_segments(
        self,
        transcript: Dict[str, Any],
        segment_ids: List[int]
    ) -> Dict[str, Any]:
        """
        Merge multiple subtitle segments into one.
        
        Args:
            transcript: Transcript with timing information
            segment_ids: List of segment indices to merge (in order)
            
        Returns:
            Updated transcript
        """
        if not transcript or "segments" not in transcript:
            raise ValueError("Invalid transcript format")
        
        if not segment_ids or len(segment_ids) < 2:
            raise ValueError("At least two segments must be specified for merging")
        
        # Validate segment IDs
        segments = transcript["segments"]
        for segment_id in segment_ids:
            if segment_id < 0 or segment_id >= len(segments):
                raise ValueError(f"Invalid segment ID: {segment_id}")
        
        # Sort segment IDs to ensure correct order
        segment_ids = sorted(segment_ids)
        
        # Check if segments are consecutive
        for i in range(len(segment_ids) - 1):
            if segment_ids[i + 1] != segment_ids[i] + 1:
                raise ValueError("Only consecutive segments can be merged")
        
        # Create a deep copy to avoid modifying the original
        transcript_copy = deepcopy(transcript)
        segments = transcript_copy["segments"]
        
        # Store original segments for history
        original_segments = [deepcopy(segments[i]) for i in segment_ids]
        
        # Create the merged segment
        start_time = segments[segment_ids[0]].get("start", 0)
        end_time = segments[segment_ids[-1]].get("end", 0)
        
        # Combine text from all segments
        texts = [segments[i].get("text", "") for i in segment_ids]
        merged_text = " ".join(texts)
        
        # Create a new merged segment
        merged_segment = {
            "start": start_time,
            "end": end_time,
            "text": merged_text
        }
        
        # Replace the first segment with the merged one
        segments[segment_ids[0]] = merged_segment
        
        # Remove the other segments (in reverse order to maintain indices)
        for i in reversed(segment_ids[1:]):
            segments.pop(i)
        
        # Create an edit operation for history
        operation = EditOperation(
            operation_type="merge",
            segment_ids=segment_ids,
            params={
                "original_segments": original_segments,
                "merged_segment": deepcopy(merged_segment)
            },
            description=f"Merge segments {', '.join(map(str, segment_ids))}"
        )
        
        # Add operation to history
        self._add_to_history(operation, transcript_copy)
        
        return transcript_copy
    
    def add_segment(
        self,
        transcript: Dict[str, Any],
        start_time: float,
        end_time: float,
        text: str,
        position: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Add a new subtitle segment.
        
        Args:
            transcript: Transcript with timing information
            start_time: Start time in seconds
            end_time: End time in seconds
            text: Subtitle text
            position: Index to insert at (None = auto-position based on time)
            
        Returns:
            Updated transcript
        """
        if not transcript or "segments" not in transcript:
            raise ValueError("Invalid transcript format")
        
        if start_time >= end_time:
            raise ValueError("Start time must be before end time")
        
        # Create a deep copy to avoid modifying the original
        transcript_copy = deepcopy(transcript)
        segments = transcript_copy["segments"]
        
        # Create the new segment
        new_segment = {
            "start": start_time,
            "end": end_time,
            "text": text
        }
        
        # Determine position based on time if not specified
        if position is None:
            position = 0
            for i, segment in enumerate(segments):
                if start_time > segment.get("start", 0):
                    position = i + 1
        
        # Validate position
        if position < 0 or position > len(segments):
            raise ValueError(f"Invalid position: {position}")
        
        # Insert the new segment
        segments.insert(position, new_segment)
        
        # Create an edit operation for history
        operation = EditOperation(
            operation_type="add",
            segment_ids=[position],
            params={
                "segment": deepcopy(new_segment)
            },
            description=f"Add segment at position {position}"
        )
        
        # Add operation to history
        self._add_to_history(operation, transcript_copy)
        
        return transcript_copy
    
    def delete_segment(
        self,
        transcript: Dict[str, Any],
        segment_id: int
    ) -> Dict[str, Any]:
        """
        Delete a subtitle segment.
        
        Args:
            transcript: Transcript with timing information
            segment_id: Index of the segment to delete
            
        Returns:
            Updated transcript
        """
        if not transcript or "segments" not in transcript:
            raise ValueError("Invalid transcript format")
        
        if segment_id < 0 or segment_id >= len(transcript["segments"]):
            raise ValueError(f"Invalid segment ID: {segment_id}")
        
        # Create a deep copy to avoid modifying the original
        transcript_copy = deepcopy(transcript)
        segments = transcript_copy["segments"]
        
        # Store the deleted segment for history
        deleted_segment = deepcopy(segments[segment_id])
        
        # Remove the segment
        segments.pop(segment_id)
        
        # Create an edit operation for history
        operation = EditOperation(
            operation_type="delete",
            segment_ids=[segment_id],
            params={
                "segment": deleted_segment
            },
            description=f"Delete segment {segment_id}"
        )
        
        # Add operation to history
        self._add_to_history(operation, transcript_copy)
        
        return transcript_copy
    
    def shift_timings(
        self,
        transcript: Dict[str, Any],
        time_shift: float,
        segment_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Shift subtitle timings by a specified amount.
        
        Args:
            transcript: Transcript with timing information
            time_shift: Amount to shift (seconds, positive or negative)
            segment_ids: List of segment indices to shift (None = all segments)
            
        Returns:
            Updated transcript
        """
        if not transcript or "segments" not in transcript:
            raise ValueError("Invalid transcript format")
        
        # Create a deep copy to avoid modifying the original
        transcript_copy = deepcopy(transcript)
        segments = transcript_copy["segments"]
        
        # Determine which segments to shift
        if segment_ids is None:
            segment_ids = list(range(len(segments)))
        else:
            # Validate segment IDs
            for segment_id in segment_ids:
                if segment_id < 0 or segment_id >= len(segments):
                    raise ValueError(f"Invalid segment ID: {segment_id}")
        
        # Store original timings for history
        original_timings = {
            i: (segments[i].get("start", 0), segments[i].get("end", 0))
            for i in segment_ids
        }
        
        # Shift segment timings
        for segment_id in segment_ids:
            segment = segments[segment_id]
            segment["start"] = max(0, segment.get("start", 0) + time_shift)
            segment["end"] = max(segment["start"] + 0.1, segment.get("end", 0) + time_shift)
        
        # Create an edit operation for history
        operation = EditOperation(
            operation_type="shift",
            segment_ids=segment_ids,
            params={
                "time_shift": time_shift,
                "original_timings": original_timings
            },
            description=f"Shift timings by {time_shift:.2f}s for {len(segment_ids)} segments"
        )
        
        # Add operation to history
        self._add_to_history(operation, transcript_copy)
        
        return transcript_copy
    
    def undo(self, transcript: Dict[str, Any]) -> Dict[str, Any]:
        """
        Undo the last operation.
        
        Args:
            transcript: Current transcript state
            
        Returns:
            Updated transcript with last operation undone, or original if no operations to undo
        """
        if self.current_position < 0:
            return transcript  # Nothing to undo
        
        # Get the previous state
        previous_state = self.history[self.current_position]
        
        # Decrement position
        self.current_position -= 1
        
        self.logger.info(f"Undo: {previous_state['operation'].description}")
        
        # Return the previous state
        return deepcopy(previous_state["transcript_before"])
    
    def redo(self, transcript: Dict[str, Any]) -> Dict[str, Any]:
        """
        Redo the last undone operation.
        
        Args:
            transcript: Current transcript state
            
        Returns:
            Updated transcript with last undone operation redone, or original if no operations to redo
        """
        if self.current_position >= len(self.history) - 1:
            return transcript  # Nothing to redo
        
        # Increment position
        self.current_position += 1
        
        # Get the state to redo
        redo_state = self.history[self.current_position]
        
        self.logger.info(f"Redo: {redo_state['operation'].description}")
        
        # Return the redo state
        return deepcopy(redo_state["transcript_after"])
    
    def can_undo(self) -> bool:
        """Check if undo is available."""
        return self.current_position >= 0
    
    def can_redo(self) -> bool:
        """Check if redo is available."""
        return self.current_position < len(self.history) - 1
    
    def get_history_summary(self) -> List[str]:
        """
        Get a summary of the operation history.
        
        Returns:
            List of operation descriptions
        """
        return [item["operation"].description for item in self.history]
    
    async def save_to_file(
        self,
        transcript: Dict[str, Any],
        output_path: str,
        format: SubtitleFormat = SubtitleFormat.SRT,
        style_name: Optional[str] = None,
        custom_style: Optional[SubtitleStyle] = None
    ) -> str:
        """
        Save the transcript to a subtitle file.
        
        Args:
            transcript: Transcript with timing information
            output_path: Path to save the subtitle file
            format: Subtitle format to use
            style_name: Name of style template to use
            custom_style: Custom style overrides
            
        Returns:
            Path to the subtitle file
        """
        return await self.subtitle_generator.generate_subtitles(
            transcript=transcript,
            output_path=output_path,
            format=format,
            style_name=style_name,
            custom_style=custom_style
        )
    
    def _add_to_history(self, operation: EditOperation, transcript_after: Dict[str, Any]) -> None:
        """
        Add an operation to the history.
        
        Args:
            operation: Edit operation
            transcript_after: Transcript state after the operation
        """
        # Get the previous state
        transcript_before = None
        if self.current_position >= 0:
            transcript_before = self.history[self.current_position]["transcript_after"]
        else:
            transcript_before = deepcopy(transcript_after)  # No previous state
        
        # Create history entry
        history_entry = {
            "operation": operation,
            "transcript_before": deepcopy(transcript_before),
            "transcript_after": deepcopy(transcript_after)
        }
        
        # If we're not at the end of the history, remove future entries
        if self.current_position < len(self.history) - 1:
            self.history = self.history[:self.current_position + 1]
        
        # Add the new entry
        self.history.append(history_entry)
        self.current_position += 1
        
        # Trim history if it exceeds max size
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
            self.current_position = len(self.history) - 1
        
        self.logger.debug(f"Added to history: {operation.description}")
        
    def analyze_subtitle_quality(self, transcript: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the quality of subtitles based on timing, length, and other factors.
        
        Args:
            transcript: Transcript with timing information
            
        Returns:
            Dictionary with quality metrics and issues
        """
        if not transcript or "segments" not in transcript:
            raise ValueError("Invalid transcript format")
        
        segments = transcript["segments"]
        
        # Initialize quality metrics
        quality = {
            "overall_score": 0.0,
            "issues": [],
            "metrics": {
                "total_segments": len(segments),
                "average_duration": 0.0,
                "average_chars_per_second": 0.0,
                "max_chars_per_second": 0.0
            }
        }
        
        if not segments:
            return quality
        
        # Track issues
        issues = []
        
        # Calculate metrics
        total_duration = 0.0
        total_chars = 0
        max_chars_per_second = 0.0
        
        # Ideal ranges
        ideal_duration_range = (0.7, 7.0)  # Seconds
        ideal_chars_per_second_range = (5, 25)  # Characters per second
        max_chars_per_line = 42
        min_gap = 0.05  # Minimum gap between subtitles
        
        # Check each segment
        for i, segment in enumerate(segments):
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            text = segment.get("text", "")
            
            # Calculate segment metrics
            duration = end_time - start_time
            chars = len(text)
            chars_per_second = chars / duration if duration > 0 else 0
            
            # Check for issues
            
            # 1. Duration issues
            if duration < ideal_duration_range[0]:
                issues.append({
                    "type": "short_duration",
                    "segment_id": i,
                    "message": f"Segment {i} is too short ({duration:.2f}s)",
                    "severity": "warning"
                })
            elif duration > ideal_duration_range[1]:
                issues.append({
                    "type": "long_duration",
                    "segment_id": i,
                    "message": f"Segment {i} is too long ({duration:.2f}s)",
                    "severity": "warning"
                })
            
            # 2. Reading speed issues
            if chars_per_second > ideal_chars_per_second_range[1]:
                issues.append({
                    "type": "fast_reading",
                    "segment_id": i,
                    "message": f"Segment {i} has too many characters per second ({chars_per_second:.1f})",
                    "severity": "error"
                })
            elif chars_per_second < ideal_chars_per_second_range[0] and chars > 10:
                issues.append({
                    "type": "slow_reading",
                    "segment_id": i,
                    "message": f"Segment {i} has too few characters per second ({chars_per_second:.1f})",
                    "severity": "info"
                })
            
            # 3. Line length issues
            lines = text.split('\n')
            for j, line in enumerate(lines):
                if len(line) > max_chars_per_line:
                    issues.append({
                        "type": "long_line",
                        "segment_id": i,
                        "message": f"Line {j+1} in segment {i} is too long ({len(line)} chars)",
                        "severity": "warning"
                    })
            
            # 4. Timing gap issues (with next segment)
            if i < len(segments) - 1:
                next_start = segments[i+1].get("start", 0)
                gap = next_start - end_time
                
                if gap < 0:
                    issues.append({
                        "type": "overlap",
                        "segment_id": i,
                        "message": f"Segment {i} overlaps with segment {i+1} by {abs(gap):.2f}s",
                        "severity": "error"
                    })
                elif gap < min_gap:
                    issues.append({
                        "type": "small_gap",
                        "segment_id": i,
                        "message": f"Gap between segments {i} and {i+1} is too small ({gap:.2f}s)",
                        "severity": "warning"
                    })
            
            # Update totals
            total_duration += duration
            total_chars += chars
            max_chars_per_second = max(max_chars_per_second, chars_per_second)
        
        # Calculate averages
        average_duration = total_duration / len(segments) if segments else 0
        average_chars_per_second = total_chars / total_duration if total_duration > 0 else 0
        
        # Update metrics
        quality["metrics"]["average_duration"] = average_duration
        quality["metrics"]["average_chars_per_second"] = average_chars_per_second
        quality["metrics"]["max_chars_per_second"] = max_chars_per_second
        
        # Calculate overall score (0-100)
        error_count = sum(1 for issue in issues if issue["severity"] == "error")
        warning_count = sum(1 for issue in issues if issue["severity"] == "warning")
        info_count = sum(1 for issue in issues if issue["severity"] == "info")
        
        # Base score
        score = 100
        
        # Deduct for issues
        score -= error_count * 10  # -10 points per error
        score -= warning_count * 3  # -3 points per warning
        score -= info_count * 1     # -1 point per info
        
        # Ensure score is in 0-100 range
        score = max(0, min(100, score))
        
        quality["overall_score"] = score
        quality["issues"] = issues
        
        return quality
    
    def auto_adjust_timings(self, transcript: Dict[str, Any]) -> Dict[str, Any]:
        """
        Automatically adjust subtitle timings to improve readability.
        
        Args:
            transcript: Transcript with timing information
            
        Returns:
            Updated transcript with adjusted timings
        """
        if not transcript or "segments" not in transcript:
            raise ValueError("Invalid transcript format")
        
        # Create a deep copy to avoid modifying the original
        transcript_copy = deepcopy(transcript)
        segments = transcript_copy["segments"]
        
        if not segments:
            return transcript_copy
        
        # Track original timings for history
        original_timings = {
            i: (segment.get("start", 0), segment.get("end", 0))
            for i, segment in enumerate(segments)
        }
        
        # Auto-adjust parameters
        min_segment_duration = 0.7  # Seconds
        max_segment_duration = 7.0  # Seconds
        target_chars_per_second = 15  # Characters per second
        min_gap = 0.2  # Minimum gap between segments
        
        # Adjust each segment
        for i, segment in enumerate(segments):
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            text = segment.get("text", "")
            
            # Calculate ideal duration based on content
            chars = len(text)
            ideal_duration = chars / target_chars_per_second
            
            # Ensure minimum and maximum duration
            ideal_duration = max(min_segment_duration, min(max_segment_duration, ideal_duration))
            
            # Calculate new end time
            new_end_time = start_time + ideal_duration
            
            # If this would overlap with the next segment, adjust
            if i < len(segments) - 1:
                next_start = segments[i+1].get("start", 0)
                if new_end_time + min_gap > next_start:
                    # Adjust the current segment's end time
                    new_end_time = next_start - min_gap
                    
                    # Ensure minimum duration
                    if new_end_time - start_time < min_segment_duration:
                        new_end_time = start_time + min_segment_duration
                        
                        # If this still overlaps, adjust the next segment's start time
                        if new_end_time + min_gap > next_start:
                            segments[i+1]["start"] = new_end_time + min_gap
                            
                            # Ensure minimum duration for next segment
                            next_end = segments[i+1].get("end", 0)
                            if next_end - (new_end_time + min_gap) < min_segment_duration:
                                segments[i+1]["end"] = new_end_time + min_gap + min_segment_duration
            
            # Update the segment's end time
            segment["end"] = new_end_time
        
        # Create an edit operation for history
        operation = EditOperation(
            operation_type="auto_adjust",
            segment_ids=list(range(len(segments))),
            params={
                "original_timings": original_timings
            },
            description="Auto-adjust subtitle timings"
        )
        
        # Add operation to history
        self._add_to_history(operation, transcript_copy)
        
        return transcript_copy 