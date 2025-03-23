import logging
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import os
import json
import subprocess
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class HighlightSegment:
    """Represents a segment selected for inclusion in highlights."""
    start: float
    end: float
    importance: float
    content_type: str  # speech, visual, action, etc.
    reason: str  # Why this segment was selected


class HighlightGenerator:
    """
    Automatically generates video highlights by identifying and extracting
    the most important segments from a video based on content analysis.
    
    Features:
    - Multimodal importance scoring
    - Adaptive highlight duration
    - Narrative structure preservation
    - Transition smoothing
    - Format-specific optimization
    """
    
    def __init__(
        self,
        enabled: bool = True,
        min_highlight_duration: float = 30.0,
        max_highlight_duration: float = 180.0,
        target_highlight_ratio: float = 0.2,
        min_segment_duration: float = 1.5,
        max_segment_duration: float = 20.0,
        narrative_preservation: bool = True,
        smoothing_enabled: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the highlight generator.
        
        Args:
            enabled: Whether highlight generation is enabled
            min_highlight_duration: Minimum duration for generated highlights
            max_highlight_duration: Maximum duration for generated highlights
            target_highlight_ratio: Target ratio of highlight to original duration
            min_segment_duration: Minimum duration for a highlight segment
            max_segment_duration: Maximum duration for a highlight segment
            narrative_preservation: Whether to preserve narrative structure
            smoothing_enabled: Whether to apply transition smoothing
            config: Additional configuration options
        """
        self.enabled = enabled
        self.min_highlight_duration = min_highlight_duration
        self.max_highlight_duration = max_highlight_duration
        self.target_highlight_ratio = target_highlight_ratio
        self.min_segment_duration = min_segment_duration
        self.max_segment_duration = max_segment_duration
        self.narrative_preservation = narrative_preservation
        self.smoothing_enabled = smoothing_enabled
        
        self.config = config or {}
        
        # Component importance weights
        self.weights = self.config.get("component_weights", {
            "speech": 0.5,
            "visual": 0.3,
            "pacing": 0.2
        })
        
        # Segment selection strategy
        self.selection_strategy = self.config.get("selection_strategy", "importance_threshold")
        
        self.logger = logging.getLogger(__name__)
    
    async def generate_highlights(
        self,
        video_path: str,
        content_analysis: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate video highlights based on content analysis.
        
        Args:
            video_path: Path to original video file
            content_analysis: Results from content analysis system
            output_path: Path to save the highlight video
            
        Returns:
            Dict containing highlight information and segments
        """
        if not self.enabled:
            self.logger.warning("Highlight generation disabled")
            return {"enabled": False}
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        try:
            self.logger.info(f"Generating highlights for video: {video_path}")
            
            # Get video duration
            duration = await self._get_video_duration(video_path)
            
            # Delegate CPU-intensive work to a thread pool
            loop = asyncio.get_event_loop()
            highlight_info = await loop.run_in_executor(
                None,
                self._generate_highlights_sync,
                video_path,
                content_analysis,
                duration
            )
            
            # Generate the actual highlight video if output path is provided
            if output_path:
                highlight_video = await self._create_highlight_video(
                    video_path, 
                    highlight_info["segments"],
                    output_path
                )
                highlight_info["highlight_video_path"] = highlight_video
            
            self.logger.info(
                f"Highlight generation complete for {video_path}. "
                f"Selected {len(highlight_info['segments'])} segments "
                f"for a total of {highlight_info['total_duration']:.2f} seconds."
            )
            
            return highlight_info
            
        except Exception as e:
            self.logger.error(f"Error generating highlights: {str(e)}")
            raise
    
    def _generate_highlights_sync(
        self,
        video_path: str,
        content_analysis: Dict[str, Any],
        duration: float
    ) -> Dict[str, Any]:
        """
        Synchronous implementation of highlight generation.
        
        Args:
            video_path: Path to the video file
            content_analysis: Content analysis results
            duration: Video duration in seconds
            
        Returns:
            Dict containing highlight information
        """
        # Calculate target highlight duration
        target_duration = self._calculate_target_duration(duration)
        
        # Build unified importance timeline
        timeline = self._build_unified_timeline(content_analysis)
        
        # Select segments for highlights
        selected_segments = self._select_highlight_segments(
            timeline, 
            target_duration,
            content_analysis
        )
        
        # Post-process segments (merge nearby segments, trim, etc.)
        processed_segments = self._post_process_segments(selected_segments)
        
        # Calculate actual highlight duration
        total_highlight_duration = sum(segment.end - segment.start for segment in processed_segments)
        
        # Format segments for output
        formatted_segments = [
            {
                "start": segment.start,
                "end": segment.end,
                "duration": segment.end - segment.start,
                "importance": segment.importance,
                "content_type": segment.content_type,
                "reason": segment.reason
            }
            for segment in processed_segments
        ]
        
        return {
            "original_duration": duration,
            "target_duration": target_duration,
            "total_duration": total_highlight_duration,
            "duration_ratio": total_highlight_duration / duration if duration > 0 else 0,
            "segments_count": len(processed_segments),
            "segments": formatted_segments
        }
    
    async def _get_video_duration(self, video_path: str) -> float:
        """Get video duration using ffprobe."""
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json",
                video_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                self.logger.error(f"FFprobe error: {stderr.decode()}")
                raise Exception(f"FFprobe failed: {stderr.decode()}")
            
            result = json.loads(stdout.decode())
            duration = float(result["format"]["duration"])
            
            return duration
            
        except Exception as e:
            self.logger.error(f"Error getting video duration: {str(e)}")
            raise
    
    def _calculate_target_duration(self, original_duration: float) -> float:
        """
        Calculate target highlight duration based on original video duration.
        
        Args:
            original_duration: Original video duration in seconds
            
        Returns:
            Target highlight duration in seconds
        """
        # Calculate based on target ratio
        target = original_duration * self.target_highlight_ratio
        
        # Clamp to min/max boundaries
        target = max(self.min_highlight_duration, min(self.max_highlight_duration, target))
        
        return target
    
    def _build_unified_timeline(self, content_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build a unified timeline with importance scores.
        
        Args:
            content_analysis: Content analysis results
            
        Returns:
            List of timeline entries with importance scores
        """
        timeline = []
        
        # Extract speech importance scores
        speech_data = content_analysis.get("speech_analysis", {})
        if "segments" in speech_data:
            for segment in speech_data["segments"]:
                timeline.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "importance": segment.get("importance", 0.5),
                    "component": "speech",
                    "content_type": segment.get("segment_type", "speech"),
                    "text": segment.get("text", "")
                })
        
        # Extract visual importance scores
        visual_data = content_analysis.get("visual_analysis", {})
        if "segments" in visual_data:
            for segment in visual_data["segments"]:
                timeline.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "importance": segment.get("interest_score", 0.5),
                    "component": "visual",
                    "content_type": segment.get("interest_type", "visual"),
                    "features": segment.get("features", {})
                })
        
        # Extract pacing analysis (for segments with good pacing)
        pacing_data = content_analysis.get("pacing_analysis", {})
        if "segments" in pacing_data:
            for segment in pacing_data["segments"]:
                # Include segments with good pacing
                if segment.get("pacing", "") == "good":
                    timeline.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "importance": 0.7,  # Good pacing is important
                        "component": "pacing",
                        "content_type": "good_pacing"
                    })
        
        return timeline
    
    def _select_highlight_segments(
        self,
        timeline: List[Dict[str, Any]],
        target_duration: float,
        content_analysis: Dict[str, Any]
    ) -> List[HighlightSegment]:
        """
        Select segments for inclusion in highlights.
        
        Args:
            timeline: Unified timeline with importance scores
            target_duration: Target highlight duration
            content_analysis: Full content analysis results
            
        Returns:
            List of selected highlight segments
        """
        if self.selection_strategy == "importance_threshold":
            return self._select_by_importance_threshold(timeline, target_duration)
        elif self.selection_strategy == "narrative_clustering":
            return self._select_by_narrative_clustering(timeline, target_duration, content_analysis)
        else:
            self.logger.warning(f"Unknown selection strategy: {self.selection_strategy}")
            return self._select_by_importance_threshold(timeline, target_duration)
    
    def _select_by_importance_threshold(
        self,
        timeline: List[Dict[str, Any]],
        target_duration: float
    ) -> List[HighlightSegment]:
        """
        Select segments based on importance threshold.
        
        Args:
            timeline: Unified timeline with importance scores
            target_duration: Target highlight duration
            
        Returns:
            List of selected highlight segments
        """
        # Sort timeline entries by importance
        sorted_entries = sorted(timeline, key=lambda x: x["importance"], reverse=True)
        
        # Initialize selected segments and current duration
        selected_segments = []
        current_duration = 0
        
        # Select segments by importance until we reach the target duration
        for entry in sorted_entries:
            segment_duration = entry["end"] - entry["start"]
            
            # Skip segments that are too short or too long
            if segment_duration < self.min_segment_duration:
                continue
                
            if segment_duration > self.max_segment_duration:
                # Could consider trimming here, but for simplicity we'll skip
                continue
            
            # Skip if we already have a segment that overlaps significantly
            if self._overlaps_with_selected(entry, selected_segments):
                continue
            
            # Add segment to selected
            highlight_segment = HighlightSegment(
                start=entry["start"],
                end=entry["end"],
                importance=entry["importance"],
                content_type=entry["component"],
                reason=f"High importance {entry['component']} content: {entry.get('content_type', 'unknown')}"
            )
            
            selected_segments.append(highlight_segment)
            current_duration += segment_duration
            
            # Stop if we've reached the target duration
            if current_duration >= target_duration:
                break
        
        # Sort selected segments by start time for chronological order
        selected_segments.sort(key=lambda x: x.start)
        
        return selected_segments
    
    def _select_by_narrative_clustering(
        self,
        timeline: List[Dict[str, Any]],
        target_duration: float,
        content_analysis: Dict[str, Any]
    ) -> List[HighlightSegment]:
        """
        Select segments based on narrative clustering.
        Groups important segments into narrative clusters.
        
        Args:
            timeline: Unified timeline with importance scores
            target_duration: Target highlight duration
            content_analysis: Full content analysis results
            
        Returns:
            List of selected highlight segments
        """
        # Group timeline into clusters
        clusters = self._cluster_timeline(timeline)
        
        # Calculate importance of each cluster
        cluster_importance = {}
        for cluster_id, entries in clusters.items():
            # Weighted average of importance scores
            total_weight = 0
            total_score = 0
            
            for entry in entries:
                weight = entry["end"] - entry["start"]
                component_weight = self.weights.get(entry["component"], 0.5)
                
                total_score += entry["importance"] * weight * component_weight
                total_weight += weight * component_weight
            
            avg_importance = total_score / total_weight if total_weight > 0 else 0
            cluster_importance[cluster_id] = avg_importance
        
        # Sort clusters by importance
        sorted_clusters = sorted(
            clusters.items(), 
            key=lambda x: cluster_importance[x[0]],
            reverse=True
        )
        
        # Select clusters until we reach target duration
        selected_segments = []
        current_duration = 0
        
        for cluster_id, entries in sorted_clusters:
            # Find cluster boundaries
            start = min(entry["start"] for entry in entries)
            end = max(entry["end"] for entry in entries)
            cluster_duration = end - start
            
            # Skip if cluster is too long
            if cluster_duration > self.max_segment_duration:
                # Could divide into sub-segments, but for simplicity we'll skip
                continue
            
            # Find the most important entry in this cluster for classification
            most_important = max(entries, key=lambda x: x["importance"])
            
            # Add cluster as a segment
            highlight_segment = HighlightSegment(
                start=start,
                end=end,
                importance=cluster_importance[cluster_id],
                content_type=most_important["component"],
                reason=f"Important narrative cluster: {most_important.get('content_type', 'unknown')}"
            )
            
            selected_segments.append(highlight_segment)
            current_duration += cluster_duration
            
            # Stop if we've reached target duration
            if current_duration >= target_duration:
                break
        
        # Sort selected segments by start time
        selected_segments.sort(key=lambda x: x.start)
        
        return selected_segments
    
    def _cluster_timeline(self, timeline: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """
        Cluster timeline entries based on temporal proximity.
        
        Args:
            timeline: Unified timeline with importance scores
            
        Returns:
            Dict mapping cluster IDs to lists of timeline entries
        """
        # Sort timeline by start time
        sorted_timeline = sorted(timeline, key=lambda x: x["start"])
        
        # Initialize clusters
        clusters = defaultdict(list)
        current_cluster = 0
        
        # Maximum gap between segments to be considered part of the same cluster
        max_gap = 3.0  # seconds
        
        if not sorted_timeline:
            return clusters
            
        # Add first entry to first cluster
        clusters[current_cluster].append(sorted_timeline[0])
        last_end = sorted_timeline[0]["end"]
        
        # Cluster remaining entries
        for entry in sorted_timeline[1:]:
            # If entry starts within max_gap of previous entry end,
            # add to current cluster
            if entry["start"] - last_end <= max_gap:
                clusters[current_cluster].append(entry)
            else:
                # Start a new cluster
                current_cluster += 1
                clusters[current_cluster].append(entry)
            
            # Update last_end to be the max of current last_end and this entry's end
            last_end = max(last_end, entry["end"])
        
        return clusters
    
    def _overlaps_with_selected(
        self, 
        entry: Dict[str, Any],
        selected_segments: List[HighlightSegment]
    ) -> bool:
        """
        Check if an entry overlaps significantly with already selected segments.
        
        Args:
            entry: Timeline entry to check
            selected_segments: List of already selected segments
            
        Returns:
            True if significant overlap exists, False otherwise
        """
        # Overlap threshold (percentage of segment that can overlap)
        overlap_threshold = 0.5
        
        for segment in selected_segments:
            # Calculate overlap
            overlap_start = max(entry["start"], segment.start)
            overlap_end = min(entry["end"], segment.end)
            
            if overlap_start < overlap_end:  # There is overlap
                overlap_duration = overlap_end - overlap_start
                entry_duration = entry["end"] - entry["start"]
                
                # If overlap is significant relative to entry size
                if overlap_duration / entry_duration > overlap_threshold:
                    return True
        
        return False
    
    def _post_process_segments(self, segments: List[HighlightSegment]) -> List[HighlightSegment]:
        """
        Post-process selected segments to improve transitions.
        
        Args:
            segments: List of selected highlight segments
            
        Returns:
            List of post-processed highlight segments
        """
        if not segments:
            return []
            
        # Sort segments by start time to ensure proper ordering
        segments.sort(key=lambda x: x.start)
        
        # Merge neighboring segments if they're close enough
        merged_segments = []
        current_segment = segments[0]
        
        for next_segment in segments[1:]:
            # Calculate gap between current and next segment
            gap = next_segment.start - current_segment.end
            
            # If gap is small, merge segments
            if gap <= 0.5:  # 0.5 second threshold for merging
                # Create new merged segment
                merged = HighlightSegment(
                    start=current_segment.start,
                    end=next_segment.end,
                    importance=max(current_segment.importance, next_segment.importance),
                    content_type=current_segment.content_type,
                    reason=f"Merged: {current_segment.reason} + {next_segment.reason}"
                )
                current_segment = merged
            else:
                # Add current segment to results and move to next
                merged_segments.append(current_segment)
                current_segment = next_segment
        
        # Add the last segment
        merged_segments.append(current_segment)
        
        # Apply padding if enabled
        if self.config.get("padding_enabled", True):
            padded_segments = []
            
            for segment in merged_segments:
                # Add padding before and after segment
                padding = self.config.get("padding_seconds", 0.5)
                
                padded = HighlightSegment(
                    start=max(0, segment.start - padding),
                    end=segment.end + padding,
                    importance=segment.importance,
                    content_type=segment.content_type,
                    reason=segment.reason
                )
                
                padded_segments.append(padded)
            
            return padded_segments
        else:
            return merged_segments
    
    async def _create_highlight_video(
        self,
        source_video: str,
        segments: List[Dict[str, Any]],
        output_path: str
    ) -> str:
        """
        Create a highlight video from selected segments.
        
        Args:
            source_video: Path to source video
            segments: List of selected segments
            output_path: Path to save the highlight video
            
        Returns:
            Path to the created highlight video
        """
        if not segments:
            self.logger.warning("No segments selected for highlights")
            return None
            
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create a temporary file for the segment list
        segments_file = f"{output_path}.segments.txt"
        
        try:
            # Write segment list in FFmpeg concat format
            with open(segments_file, 'w') as f:
                for i, segment in enumerate(segments):
                    # Write each segment entry
                    f.write(f"file '{source_video}'\n")
                    f.write(f"inpoint {segment['start']:.6f}\n")
                    f.write(f"outpoint {segment['end']:.6f}\n")
            
            # Build FFmpeg command for creating the highlight video
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output files without asking
                "-f", "concat", 
                "-safe", "0",  # Allow absolute paths
                "-i", segments_file,
                "-c", "copy",  # Copy codecs without re-encoding for speed
                output_path
            ]
            
            # If smoothing is enabled, use more complex command
            if self.smoothing_enabled:
                # Use more complex command with crossfades
                # This is a simplified version - real implementation would be more complex
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", segments_file,
                    "-filter_complex", "[0:v]fps=fps=30,format=yuv420p[v]",
                    "-map", "[v]",
                    "-map", "0:a",
                    "-c:v", "libx264",
                    "-c:a", "aac",
                    "-b:a", "128k",
                    output_path
                ]
            
            # Execute FFmpeg command
            self.logger.info(f"Running FFmpeg to create highlight video: {' '.join(cmd)}")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                self.logger.error(f"FFmpeg error: {stderr.decode()}")
                raise Exception(f"Error creating highlight video: {stderr.decode()}")
            
            self.logger.info(f"Successfully created highlight video at {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error creating highlight video: {str(e)}")
            raise
        finally:
            # Clean up temporary files
            if os.path.exists(segments_file):
                os.remove(segments_file) 