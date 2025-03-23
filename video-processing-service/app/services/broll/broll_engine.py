"""
B-Roll Engine for intelligent video content enhancement.

This module orchestrates the entire B-Roll insertion process, combining content
analysis, scene detection, and stock footage integration to intelligently enhance
videos with relevant B-Roll footage.
"""

import logging
import os
import json
import tempfile
import asyncio
import time
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from app.services.broll.content_analyzer import ContentAnalyzer
from app.services.broll.scene_detector import SceneDetector
from app.services.broll.stock_integration import StockFootageProvider
from app.services.broll.semantic_matcher import SemanticMatcher

logger = logging.getLogger(__name__)

class BRollEngine:
    """
    Main engine for intelligent B-Roll insertion and management.
    
    This class combines content analysis, scene detection, and stock footage
    integration to provide a comprehensive solution for enhancing videos
    with relevant B-Roll footage.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the BRollEngine.
        
        Args:
            config: Configuration options for the B-Roll engine
        """
        self.config = config or {}
        
        # Initialize components
        self.content_analyzer = ContentAnalyzer(self.config.get('content_analyzer'))
        self.scene_detector = SceneDetector(self.config.get('scene_detector'))
        self.stock_provider = StockFootageProvider(self.config.get('stock_provider'))
        self.semantic_matcher = SemanticMatcher(self.config.get('semantic_matcher'))
        
        # Set default parameters
        self.min_clip_duration = self.config.get('min_clip_duration', 2.0)  # seconds
        self.max_clip_duration = self.config.get('max_clip_duration', 8.0)  # seconds
        self.b_roll_density = self.config.get('b_roll_density', 0.3)  # percentage of content to cover with b-roll
        self.ffmpeg_path = self.config.get('ffmpeg_path', 'ffmpeg')
        self.output_dir = self.config.get('output_dir', os.path.join(tempfile.gettempdir(), 'broll_output'))
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Theme consistency tracker
        self.theme_tracker = []
    
    async def close(self):
        """Close any open resources."""
        await self.stock_provider.close()
    
    async def analyze_video_content(
        self, 
        video_path: str, 
        transcript: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze video content to identify B-Roll opportunities.
        
        Args:
            video_path: Path to the video file
            transcript: Transcript with timing information
            
        Returns:
            Dict containing analysis results
        """
        # Analyze transcript content
        content_analysis = await self.content_analyzer.analyze_transcript(transcript)
        
        # Detect scenes and insertion points
        scene_analysis = await self.scene_detector.find_broll_insertion_points(video_path, transcript)
        
        # Combine results
        analysis_results = {
            'content_analysis': content_analysis,
            'scene_analysis': scene_analysis,
            'video_path': video_path,
            'timestamp': time.time()
        }
        
        return analysis_results
    
    async def generate_b_roll_suggestions(
        self, 
        analysis_results: Dict[str, Any],
        max_suggestions: int = 3,
        provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate B-Roll suggestions based on analysis results.
        
        Args:
            analysis_results: Results from analyze_video_content
            max_suggestions: Maximum number of suggestions per insertion point
            provider: Stock footage provider to use
            
        Returns:
            Dict containing B-Roll suggestions
        """
        content_analysis = analysis_results.get('content_analysis', {})
        scene_analysis = analysis_results.get('scene_analysis', {})
        
        # Get insertion points
        insertion_points = scene_analysis.get('insertion_points', [])
        
        # Get B-Roll suggestions from stock provider
        stock_suggestions = await self.stock_provider.get_suggestions_for_content(
            content_analysis, 
            max_results=max_suggestions,
            provider=provider
        )
        
        # Combine insertion points with stock suggestions
        b_roll_plan = []
        
        for i, point in enumerate(insertion_points):
            # Get text from the insertion point
            segment_text = point.get('segment_text', '')
            segment_start = point.get('segment_start', 0)
            segment_end = point.get('segment_end', 0)
            
            # Analyze this specific segment to find the best footage
            segment_data = {
                'start': segment_start,
                'end': segment_end,
                'text': segment_text
            }
            
            segment_analysis = await self.content_analyzer._analyze_segment(segment_data)
            
            # Find footage for the segment's topics
            topics = segment_analysis.get('topics', [])
            actions = segment_analysis.get('actions', [])
            
            search_terms = topics + actions
            if not search_terms and segment_text:
                # Use the first few words if no specific topics found
                words = segment_text.split()[:3]
                search_terms = [' '.join(words)]
            
            # Get footage specifically for this insertion point
            footage_options = []
            
            for term in search_terms[:2]:  # Use top 2 search terms
                term_footage = await self.stock_provider.find_footage_for_topic(
                    term,
                    options={'per_page': max_suggestions},
                    provider=provider
                )
                
                # Filter out duplicates
                for footage in term_footage:
                    if not any(f.get('id') == footage.get('id') for f in footage_options):
                        footage_options.append(footage)
                
                if len(footage_options) >= max_suggestions:
                    break
            
            # Create insertion plan
            insertion_plan = {
                'timestamp': point.get('timestamp', 0),
                'segment_start': segment_start,
                'segment_end': segment_end,
                'segment_text': segment_text,
                'topics': topics,
                'actions': actions,
                'score': point.get('score', 0),
                'type': point.get('type', ''),
                'recommended_duration': point.get('recommended_duration', 5.0),
                'footage_options': footage_options[:max_suggestions]
            }
            
            b_roll_plan.append(insertion_plan)
        
        # Sort plan by timestamp
        b_roll_plan.sort(key=lambda x: x['timestamp'])
        
        # Apply visual theme consistency
        b_roll_plan = self._apply_theme_consistency(b_roll_plan)
        
        # Apply timing optimization
        b_roll_plan = self._optimize_timing(b_roll_plan)
        
        # Apply semantic enhancement if transcript segments are available
        if 'segments' in content_analysis:
            transcript_segments = content_analysis.get('segments', [])
            
            # Create suggestions object with the b_roll_plan
            suggestions = {
                'b_roll_plan': b_roll_plan,
                'stock_suggestions': stock_suggestions,
                'analysis_results': analysis_results
            }
            
            # Enhance with semantic matching
            try:
                enhanced_suggestions = await self.semantic_matcher.enhance_broll_suggestions(
                    suggestions, transcript_segments
                )
                
                # Check if enhancement was successful
                if enhanced_suggestions and 'b_roll_plan' in enhanced_suggestions:
                    b_roll_plan = enhanced_suggestions.get('b_roll_plan', b_roll_plan)
                    
                    # Add theme consistency info
                    theme_consistency = enhanced_suggestions.get('theme_consistency', {})
                    if theme_consistency:
                        suggestions['theme_consistency'] = theme_consistency
            except Exception as e:
                logger.warning(f"Error in semantic enhancement: {str(e)}")
        
        # Generate final suggestions
        suggestions = {
            'b_roll_plan': b_roll_plan,
            'stock_suggestions': stock_suggestions,
            'analysis_results': analysis_results
        }
        
        return suggestions
    
    def _apply_theme_consistency(
        self, 
        b_roll_plan: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply visual theme consistency to the B-Roll plan.
        
        Args:
            b_roll_plan: List of B-Roll insertion plans
            
        Returns:
            Updated B-Roll plan with consistent visual themes
        """
        if not b_roll_plan:
            return []
        
        # Track visual themes
        current_themes = {}
        
        # Apply consistency to each insertion point
        for i, insertion in enumerate(b_roll_plan):
            footage_options = insertion.get('footage_options', [])
            
            if not footage_options:
                continue
            
            # Extract themes from footage
            insertion_themes = {}
            
            for footage in footage_options:
                # Extract color information if available
                if 'colors' in footage:
                    for color in footage.get('colors', []):
                        color_hex = color.get('hex', '')
                        if color_hex:
                            if color_hex not in insertion_themes:
                                insertion_themes[color_hex] = 0
                            insertion_themes[color_hex] += 1
                
                # Extract tags as themes
                for tag in footage.get('tags', []):
                    if tag not in insertion_themes:
                        insertion_themes[tag] = 0
                    insertion_themes[tag] += 1
            
            # Update global theme tracker
            for theme, count in insertion_themes.items():
                if theme not in current_themes:
                    current_themes[theme] = 0
                current_themes[theme] += count
            
            # Add theme information to insertion
            insertion['visual_themes'] = insertion_themes
            
            # Sort footage options to prioritize consistent themes
            if i > 0 and current_themes:
                def theme_consistency_score(footage):
                    score = 0
                    # Give points for tags that match current themes
                    for tag in footage.get('tags', []):
                        if tag in current_themes:
                            score += current_themes[tag]
                    return score
                
                # Sort footage options by theme consistency score
                footage_options.sort(key=theme_consistency_score, reverse=True)
                insertion['footage_options'] = footage_options
            
            # Update the plan
            b_roll_plan[i] = insertion
        
        return b_roll_plan
    
    def _optimize_timing(
        self, 
        b_roll_plan: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Optimize B-Roll timing for better flow.
        
        Args:
            b_roll_plan: List of B-Roll insertion plans
            
        Returns:
            Updated B-Roll plan with optimized timing
        """
        if not b_roll_plan:
            return []
        
        # Check for overlapping insertions and adjust durations
        for i in range(len(b_roll_plan) - 1):
            current = b_roll_plan[i]
            next_insertion = b_roll_plan[i + 1]
            
            current_ts = current.get('timestamp', 0)
            current_duration = current.get('recommended_duration', 5.0)
            next_ts = next_insertion.get('timestamp', 0)
            
            # If there's an overlap, adjust the duration
            if current_ts + current_duration > next_ts:
                # Reduce current duration to avoid overlap
                adjusted_duration = max(self.min_clip_duration, next_ts - current_ts - 0.5)
                current['recommended_duration'] = adjusted_duration
                b_roll_plan[i] = current
        
        # Ensure minimum spacing between clips
        min_spacing = 2.0  # seconds
        
        for i in range(len(b_roll_plan) - 1):
            current = b_roll_plan[i]
            next_insertion = b_roll_plan[i + 1]
            
            current_ts = current.get('timestamp', 0)
            current_duration = current.get('recommended_duration', 5.0)
            next_ts = next_insertion.get('timestamp', 0)
            
            # If clips are too close together
            if next_ts - (current_ts + current_duration) < min_spacing:
                # Check which clip has lower score
                current_score = current.get('score', 0)
                next_score = next_insertion.get('score', 0)
                
                if current_score < next_score:
                    # Remove current if it's less important
                    current['skip'] = True
                else:
                    # Remove next if it's less important
                    next_insertion['skip'] = True
        
        # Filter out skipped insertions
        filtered_plan = [insertion for insertion in b_roll_plan if not insertion.get('skip', False)]
        
        return filtered_plan
    
    async def generate_edit_suggestions(
        self, 
        analysis_results: Dict[str, Any],
        suggestions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate edit suggestions for B-Roll insertion.
        
        Args:
            analysis_results: Results from analyze_video_content
            suggestions: B-Roll suggestions
            
        Returns:
            Dict containing edit suggestions
        """
        b_roll_plan = suggestions.get('b_roll_plan', [])
        
        # Generate edit decision list
        edl = []
        
        # Get video information
        video_path = analysis_results.get('video_path', '')
        video_info = analysis_results.get('scene_analysis', {}).get('scenes', {}).get('video_info', {})
        video_duration = video_info.get('duration', 0)
        
        if not video_duration:
            logger.error("Unable to get video duration")
            return {"error": "Unable to get video duration", "edl": []}
        
        # Generate EDL with alternating main video and B-Roll clips
        current_time = 0.0
        
        for insertion in b_roll_plan:
            timestamp = insertion.get('timestamp', 0)
            duration = min(insertion.get('recommended_duration', 5.0), self.max_clip_duration)
            
            # Select best footage option
            footage_options = insertion.get('footage_options', [])
            selected_footage = footage_options[0] if footage_options else None
            
            if not selected_footage:
                continue
            
            # Add main video clip up to insertion point
            if timestamp > current_time:
                edl.append({
                    'type': 'main_video',
                    'source': video_path,
                    'start_time': current_time,
                    'end_time': timestamp,
                    'output_start': current_time,
                    'duration': timestamp - current_time
                })
            
            # Add B-Roll clip
            edl.append({
                'type': 'b_roll',
                'source': selected_footage.get('download_url', ''),
                'source_id': selected_footage.get('id', ''),
                'provider': selected_footage.get('provider', ''),
                'start_time': 0,  # Start from beginning of B-Roll clip
                'end_time': duration,
                'output_start': timestamp,
                'duration': duration,
                'topics': insertion.get('topics', []),
                'segment_text': insertion.get('segment_text', '')
            })
            
            # Update current time
            current_time = timestamp + duration
        
        # Add final main video clip if needed
        if current_time < video_duration:
            edl.append({
                'type': 'main_video',
                'source': video_path,
                'start_time': current_time,
                'end_time': video_duration,
                'output_start': current_time,
                'duration': video_duration - current_time
            })
        
        # Generate edit suggestions
        edit_suggestions = {
            'edl': edl,
            'video_path': video_path,
            'output_duration': video_duration,
            'b_roll_count': sum(1 for clip in edl if clip.get('type') == 'b_roll'),
            'b_roll_time': sum(clip.get('duration', 0) for clip in edl if clip.get('type') == 'b_roll')
        }
        
        return edit_suggestions
    
    async def download_b_roll_previews(
        self, 
        edit_suggestions: Dict[str, Any],
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Download B-Roll previews for the suggested edits.
        
        Args:
            edit_suggestions: Edit suggestions from generate_edit_suggestions
            output_dir: Directory to save downloaded previews
            
        Returns:
            Updated edit suggestions with local file paths
        """
        output_dir = output_dir or os.path.join(self.output_dir, f"broll_previews_{int(time.time())}")
        os.makedirs(output_dir, exist_ok=True)
        
        edl = edit_suggestions.get('edl', [])
        updated_edl = []
        
        # Download previews for each B-Roll clip
        for clip in edl:
            if clip.get('type') == 'b_roll':
                source_url = clip.get('source', '')
                source_id = clip.get('source_id', '')
                provider = clip.get('provider', '')
                
                if source_url:
                    # Generate output path
                    file_name = f"broll_{source_id}_{provider}.mp4"
                    output_path = os.path.join(output_dir, file_name)
                    
                    # Download preview
                    local_path = await self.stock_provider.download_preview(source_url, output_path, provider)
                    
                    if local_path:
                        # Update clip with local path
                        clip['local_path'] = local_path
                    else:
                        logger.warning(f"Failed to download preview for {source_id} from {provider}")
            
            updated_edl.append(clip)
        
        # Update edit suggestions
        edit_suggestions['edl'] = updated_edl
        edit_suggestions['preview_dir'] = output_dir
        
        return edit_suggestions
    
    async def generate_preview_video(
        self, 
        edit_suggestions: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a preview video with B-Roll insertions.
        
        Args:
            edit_suggestions: Edit suggestions from download_b_roll_previews
            output_path: Path to save the preview video
            
        Returns:
            Path to the generated preview video
        """
        if not output_path:
            output_dir = os.path.join(self.output_dir, f"preview_{int(time.time())}")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "preview_with_broll.mp4")
        
        edl = edit_suggestions.get('edl', [])
        
        if not edl:
            logger.error("No edit decision list available")
            return None
        
        # Create a temporary EDL file
        temp_dir = tempfile.gettempdir()
        edl_path = os.path.join(temp_dir, f"edl_{int(time.time())}.txt")
        
        try:
            # Write EDL file for ffmpeg
            with open(edl_path, 'w') as f:
                for i, clip in enumerate(edl):
                    clip_type = clip.get('type', '')
                    source = clip.get('local_path', '') if clip_type == 'b_roll' else clip.get('source', '')
                    start_time = clip.get('start_time', 0)
                    duration = clip.get('duration', 0)
                    
                    if not source:
                        logger.warning(f"Skipping clip {i} - no source file")
                        continue
                    
                    f.write(f"file '{source}'\n")
                    f.write(f"inpoint {start_time}\n")
                    f.write(f"outpoint {start_time + duration}\n")
            
            # Generate preview video using ffmpeg
            cmd = [
                self.ffmpeg_path,
                '-y',  # Overwrite output file if it exists
                '-f', 'concat',
                '-safe', '0',
                '-i', edl_path,
                '-c', 'copy',
                output_path
            ]
            
            # Run ffmpeg
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Error generating preview video: {stderr.decode()}")
                return None
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating preview video: {str(e)}")
            return None
        finally:
            # Clean up EDL file
            if os.path.exists(edl_path):
                os.remove(edl_path)
    
    async def process_video(
        self, 
        video_path: str, 
        transcript: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a video to generate B-Roll suggestions and preview.
        
        Args:
            video_path: Path to the video file
            transcript: Transcript with timing information
            options: Processing options
            
        Returns:
            Dict containing results of the processing
        """
        options = options or {}
        output_dir = options.get('output_dir', os.path.join(self.output_dir, f"output_{int(time.time())}"))
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Analyze video content
        logger.info("Analyzing video content...")
        analysis_results = await self.analyze_video_content(video_path, transcript)
        
        # Step 2: Generate B-Roll suggestions
        logger.info("Generating B-Roll suggestions...")
        suggestions = await self.generate_b_roll_suggestions(
            analysis_results,
            max_suggestions=options.get('max_suggestions', 3),
            provider=options.get('provider')
        )
        
        # Enhanced semantic matching for final results
        use_semantic_matching = options.get('use_semantic_matching', True)
        if use_semantic_matching and 'segments' in transcript:
            logger.info("Applying advanced semantic matching...")
            try:
                # Perform deep semantic matching between speech and visual content
                semantic_insights = await self.semantic_matcher.match_concepts_to_transcript(
                    transcript.get('segments', [])
                )
                
                # Add semantic insights to results
                suggestions['semantic_insights'] = semantic_insights
            except Exception as e:
                logger.error(f"Error in semantic matching: {str(e)}")
        
        # Step 3: Generate edit suggestions
        logger.info("Generating edit suggestions...")
        edit_suggestions = await self.generate_edit_suggestions(
            analysis_results,
            suggestions
        )
        
        # Step 4: Download B-Roll previews
        logger.info("Downloading B-Roll previews...")
        edit_suggestions = await self.download_b_roll_previews(
            edit_suggestions,
            output_dir=os.path.join(output_dir, "previews")
        )
        
        # Step 5: Generate preview video (if requested)
        preview_path = None
        if options.get('generate_preview', True):
            logger.info("Generating preview video...")
            preview_path = await self.generate_preview_video(
                edit_suggestions,
                output_path=os.path.join(output_dir, "preview.mp4")
            )
        
        # Save results
        results = {
            'analysis_results': analysis_results,
            'suggestions': suggestions,
            'edit_suggestions': edit_suggestions,
            'preview_path': preview_path,
            'output_dir': output_dir
        }
        
        # Save results to JSON
        results_path = os.path.join(output_dir, "broll_results.json")
        with open(results_path, 'w') as f:
            # Create a serializable version of the results
            serializable_results = {
                'video_path': video_path,
                'output_dir': output_dir,
                'preview_path': preview_path,
                'b_roll_count': edit_suggestions.get('b_roll_count', 0),
                'b_roll_time': edit_suggestions.get('b_roll_time', 0),
                'edl': edit_suggestions.get('edl', []),
                'timestamp': time.time()
            }
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Processing complete. Results saved to {results_path}")
        
        return results 