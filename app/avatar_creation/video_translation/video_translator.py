"""
Video Translation Module

This module provides functionality for translating videos from one language to another,
with support for multiple features including:
- Multi-language support
- Context-aware translation
- Terminology preservation
- Script timing preservation
- Visual speech synthesis
- Voice translation with characteristic preservation
"""

import os
import logging
import tempfile
from typing import Dict, List, Optional, Union, Tuple, Any

# Import translator components
from app.avatar_creation.video_translation.translator import ContextAwareTranslator, TranslationOptions
from app.avatar_creation.video_translation.terminology import TerminologyManager
from app.avatar_creation.video_translation.timing import ScriptTimingPreserver
from app.avatar_creation.video_translation.visual_speech import VisualSpeechSynthesizer
from app.avatar_creation.video_translation.voice.voice_translator import VoiceTranslator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VideoTranslator:
    """
    Main class for translating videos from one language to another.
    
    This class integrates various components to provide a comprehensive
    video translation solution with multiple advanced features.
    """
    
    def __init__(
        self,
        context_model_path: Optional[str] = None,
        terminology_db_path: Optional[str] = None,
        visual_speech_path: Optional[str] = None,
        voice_translator: Optional[VoiceTranslator] = None,
        use_gpu: bool = True
    ):
        """
        Initialize the VideoTranslator with various components.
        
        Args:
            context_model_path: Path to the context-aware translation model
            terminology_db_path: Path to the terminology database
            visual_speech_path: Path to the visual speech synthesis model
            voice_translator: Voice translator instance or None to create a default one
            use_gpu: Whether to use GPU acceleration when available
        """
        self.device = "cuda" if use_gpu and self._is_gpu_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.context_translator = ContextAwareTranslator(
            model_dir=context_model_path or "models/translation",
            device=self.device,
            use_gpu=use_gpu
        )
        
        self.terminology_manager = TerminologyManager(
            db_path=terminology_db_path or "data/terminology.db"
        )
        
        self.timing_preserver = ScriptTimingPreserver()
        
        self.visual_speech_synthesizer = VisualSpeechSynthesizer(
            model_path=visual_speech_path or "models/visual_speech",
            device=self.device
        ) if visual_speech_path else None
        
        # Initialize voice translator if provided or create a default one
        if voice_translator:
            self.voice_translator = voice_translator
        else:
            self.voice_translator = VoiceTranslator(
                voice_model_path="models/voice/translator_model",
                emotion_model_path="models/emotion/transfer_model",
                prosody_model_path="models/prosody/model_weights",
                device=self.device
            )
    
    def _is_gpu_available(self) -> bool:
        """Check if GPU is available for computation."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def translate_script(
        self,
        script: Union[str, List[Dict]],
        source_lang: str,
        target_lang: str,
        preserve_terminology: bool = True,
        adjust_timing: bool = True
    ) -> List[Dict]:
        """
        Translate a video script from source language to target language.
        
        Args:
            script: Either a string containing the script or a list of segment dictionaries
            source_lang: Source language code
            target_lang: Target language code
            preserve_terminology: Whether to preserve terminology during translation
            adjust_timing: Whether to adjust timing for translated segments
            
        Returns:
            List of translated segment dictionaries with timing information
        """
        logger.info(f"Translating script from {source_lang} to {target_lang}")
        
        # Convert string script to segments if needed
        segments = self._parse_script(script) if isinstance(script, str) else script
        
        # Extract terms to preserve if needed
        terms_to_preserve = {}
        if preserve_terminology:
            terms_to_preserve = self.terminology_manager.extract_terms(
                " ".join([s.get("text", "") for s in segments]),
                source_lang, target_lang
            )
            logger.info(f"Extracted {len(terms_to_preserve)} terms to preserve")
        
        # Translate each segment with context
        translated_segments = []
        for i, segment in enumerate(segments):
            # Get context from surrounding segments
            context_before = [s.get("text", "") for s in segments[max(0, i-2):i]]
            context_after = [s.get("text", "") for s in segments[i+1:min(len(segments), i+3)]]
            
            # Create translation options
            options = TranslationOptions(
                preserve_technical_terms=preserve_terminology,
                maintain_original_timing=adjust_timing,
                context_window_size=3
            )
            
            # Translate with context
            translation, metadata = self.context_translator.translate_with_context(
                text=segment.get("text", ""),
                source_lang=source_lang,
                target_lang=target_lang,
                context=context_before + context_after,
                options=options
            )
            
            # Create translated segment
            translated_segment = segment.copy()
            translated_segment["text"] = translation
            translated_segment["original_text"] = segment.get("text", "")
            translated_segment["translation_metadata"] = metadata
            
            translated_segments.append(translated_segment)
        
        # Adjust timing if needed
        if adjust_timing:
            translated_segments = self.timing_preserver.adjust_timing(
                translated_segments, source_lang, target_lang
            )
            logger.info("Adjusted timing for translated segments")
        
        return translated_segments
    
    def translate_audio(
        self,
        audio_path: str,
        segments: List[Dict],
        source_lang: str,
        target_lang: str,
        output_path: str,
        preserve_voice_characteristics: bool = True,
        preserve_emotions: bool = True,
        preserve_prosody: bool = True
    ) -> str:
        """
        Translate audio from source language to target language.
        
        Args:
            audio_path: Path to the source audio file
            segments: List of segment dictionaries with text and timing
            source_lang: Source language code
            target_lang: Target language code
            output_path: Path to save the translated audio
            preserve_voice_characteristics: Whether to preserve voice characteristics
            preserve_emotions: Whether to preserve emotions
            preserve_prosody: Whether to preserve prosody
            
        Returns:
            Path to the translated audio file
        """
        logger.info(f"Translating audio from {source_lang} to {target_lang}")
        
        # Extract original text from segments
        original_text = " ".join([segment.get("original_text", "") for segment in segments 
                                if "original_text" in segment])
        
        # Extract translated text from segments
        translated_text = " ".join([segment.get("text", "") for segment in segments])
        
        # Check if segments contain speaker information
        has_speakers = any("speaker_id" in segment for segment in segments)
        
        if has_speakers:
            # Process multi-speaker translation
            speaker_segments = []
            for segment in segments:
                if "speaker_id" in segment:
                    speaker_segments.append({
                        "speaker_id": segment.get("speaker_id"),
                        "start_time": segment.get("start_time", 0),
                        "end_time": segment.get("end_time", 0),
                        "text": segment.get("original_text", "")
                    })
            
            result_path = self.voice_translator.process_multi_speaker(
                audio_path, speaker_segments, original_text, translated_text,
                output_path, source_lang, target_lang
            )
        else:
            # Process single-speaker translation
            result_path = self.voice_translator.translate_voice(
                audio_path, original_text, translated_text, output_path,
                source_lang, target_lang
            )
        
        logger.info(f"Audio translation completed: {result_path}")
        return result_path
    
    def synthesize_visual_speech(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        target_lang: str
    ) -> str:
        """
        Synthesize visual speech for the translated audio.
        
        Args:
            video_path: Path to the source video
            audio_path: Path to the translated audio
            output_path: Path to save the result video
            target_lang: Target language code
            
        Returns:
            Path to the resulting video with synthesized visual speech
        """
        if self.visual_speech_synthesizer is None:
            logger.warning("Visual speech synthesizer is not initialized, skipping synthesis")
            return video_path
        
        logger.info("Synthesizing visual speech")
        
        result_path = self.visual_speech_synthesizer.synthesize_speech(
            video_path=video_path,
            audio_path=audio_path,
            output_path=output_path,
            language=target_lang
        )
        
        logger.info(f"Visual speech synthesis completed: {result_path}")
        return result_path
    
    def translate_video(
        self,
        input_path: str,
        output_path: str,
        source_lang: str,
        target_lang: str,
        subtitles_path: Optional[str] = None,
        audio_only: bool = False,
        preserve_terminology: bool = True,
        adjust_timing: bool = True,
        preserve_voice: bool = True,
        preserve_emotions: bool = True,
        preserve_prosody: bool = True
    ) -> str:
        """
        Translate a video from source language to target language.
        
        This method performs the following steps:
        1. Extract audio and subtitles from the video
        2. Translate the script/subtitles
        3. Translate the audio while preserving characteristics
        4. Synthesize visual speech to match the translated audio
        5. Combine everything into the final translated video
        
        Args:
            input_path: Path to the input video file
            output_path: Path to save the translated video
            source_lang: Source language code
            target_lang: Target language code
            subtitles_path: Optional path to subtitles file, extracted from video if None
            audio_only: Whether to translate only the audio (no visual speech synthesis)
            preserve_terminology: Whether to preserve terminology during translation
            adjust_timing: Whether to adjust timing for translated segments
            preserve_voice: Whether to preserve voice characteristics
            preserve_emotions: Whether to preserve emotions
            preserve_prosody: Whether to preserve prosody
            
        Returns:
            Path to the translated video file
        """
        logger.info(f"Translating video from {source_lang} to {target_lang}")
        
        # Create temp directory for intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract audio and subtitles if needed
            audio_path = os.path.join(temp_dir, "audio.wav")
            self._extract_audio(input_path, audio_path)
            
            if subtitles_path is None:
                subtitles_path = os.path.join(temp_dir, "subtitles.srt")
                self._extract_subtitles(input_path, subtitles_path)
            
            # Parse subtitles into segments
            segments = self._parse_subtitles(subtitles_path)
            
            # Translate script
            translated_segments = self.translate_script(
                segments, source_lang, target_lang,
                preserve_terminology=preserve_terminology,
                adjust_timing=adjust_timing
            )
            
            # Generate translated subtitles
            translated_subtitles_path = os.path.join(temp_dir, "translated_subtitles.srt")
            self._generate_subtitles(translated_segments, translated_subtitles_path)
            
            # Translate audio
            translated_audio_path = os.path.join(temp_dir, "translated_audio.wav")
            self.translate_audio(
                audio_path, translated_segments, source_lang, target_lang,
                translated_audio_path,
                preserve_voice_characteristics=preserve_voice,
                preserve_emotions=preserve_emotions,
                preserve_prosody=preserve_prosody
            )
            
            # If audio only, just combine audio with original video
            if audio_only:
                self._combine_audio_video(input_path, translated_audio_path, output_path)
                return output_path
            
            # Synthesize visual speech
            result_path = self.synthesize_visual_speech(
                input_path, translated_audio_path, output_path, target_lang
            )
            
            # Add subtitles to the final video
            self._add_subtitles(result_path, translated_subtitles_path, output_path)
            
            logger.info(f"Video translation completed: {output_path}")
            return output_path
    
    def _parse_script(self, script: str) -> List[Dict]:
        """Parse a text script into segments."""
        # Simple parsing logic - split by lines and assign timestamps
        lines = [line.strip() for line in script.split('\n') if line.strip()]
        segments = []
        
        for i, line in enumerate(lines):
            segment = {
                "id": i,
                "text": line,
                "start_time": i * 3.0,  # Simple placeholder timing
                "end_time": (i + 1) * 3.0
            }
            segments.append(segment)
        
        return segments
    
    def _parse_subtitles(self, subtitles_path: str) -> List[Dict]:
        """Parse subtitles file into segments."""
        # Placeholder for subtitle parsing logic
        # In a real implementation, this would parse SRT or other subtitle formats
        segments = []
        
        # Simple placeholder implementation
        try:
            with open(subtitles_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse SRT format (simplified)
            blocks = content.strip().split('\n\n')
            for block in blocks:
                lines = block.split('\n')
                if len(lines) >= 3:
                    try:
                        segment_id = int(lines[0])
                        time_str = lines[1]
                        text = ' '.join(lines[2:])
                        
                        # Parse time format "00:00:00,000 --> 00:00:00,000"
                        times = time_str.split(' --> ')
                        if len(times) == 2:
                            start_time = self._parse_time(times[0])
                            end_time = self._parse_time(times[1])
                            
                            segment = {
                                "id": segment_id,
                                "text": text,
                                "start_time": start_time,
                                "end_time": end_time
                            }
                            segments.append(segment)
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Error parsing subtitle block: {e}")
                        continue
        except Exception as e:
            logger.error(f"Error parsing subtitles: {e}")
            # Create a dummy segment if parsing fails
            segments.append({
                "id": 1,
                "text": "Error parsing subtitles",
                "start_time": 0.0,
                "end_time": 5.0
            })
        
        return segments
    
    def _parse_time(self, time_str: str) -> float:
        """Parse time string into seconds."""
        # Format: "00:00:00,000" to seconds
        parts = time_str.replace(',', '.').split(':')
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        return 0.0
    
    def _generate_subtitles(self, segments: List[Dict], output_path: str) -> None:
        """Generate subtitles file from segments."""
        # Placeholder for subtitle generation logic
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for segment in segments:
                    # Format time as "00:00:00,000"
                    start_time_str = self._format_time(segment.get("start_time", 0))
                    end_time_str = self._format_time(segment.get("end_time", 0))
                    
                    # Write SRT format
                    f.write(f"{segment.get('id', 0)}\n")
                    f.write(f"{start_time_str} --> {end_time_str}\n")
                    f.write(f"{segment.get('text', '')}\n\n")
        except Exception as e:
            logger.error(f"Error generating subtitles: {e}")
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into time string."""
        # Format seconds to "00:00:00,000"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')
    
    def _extract_audio(self, video_path: str, output_path: str) -> None:
        """Extract audio from video."""
        # Placeholder for audio extraction logic
        # In a real implementation, this would use a library like ffmpeg
        logger.info(f"Extracting audio from {video_path} to {output_path}")
        # Simulate extraction for demonstration purposes
        try:
            with open(output_path, 'w') as f:
                f.write("Placeholder for extracted audio")
            logger.info("Audio extraction successful (simulated)")
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
    
    def _extract_subtitles(self, video_path: str, output_path: str) -> None:
        """Extract subtitles from video."""
        # Placeholder for subtitle extraction logic
        logger.info(f"Extracting subtitles from {video_path} to {output_path}")
        # Simulate extraction for demonstration purposes
        try:
            with open(output_path, 'w') as f:
                f.write("1\n00:00:00,000 --> 00:00:03,000\nPlaceholder subtitle text\n\n")
            logger.info("Subtitle extraction successful (simulated)")
        except Exception as e:
            logger.error(f"Error extracting subtitles: {e}")
    
    def _combine_audio_video(self, video_path: str, audio_path: str, output_path: str) -> None:
        """Combine audio with video."""
        # Placeholder for audio-video combination logic
        logger.info(f"Combining video {video_path} with audio {audio_path} to {output_path}")
        # Simulate combination for demonstration purposes
        try:
            with open(output_path, 'w') as f:
                f.write("Placeholder for combined video")
            logger.info("Audio-video combination successful (simulated)")
        except Exception as e:
            logger.error(f"Error combining audio and video: {e}")
    
    def _add_subtitles(self, video_path: str, subtitles_path: str, output_path: str) -> None:
        """Add subtitles to video."""
        # Placeholder for subtitle addition logic
        logger.info(f"Adding subtitles {subtitles_path} to video {video_path}")
        # If input and output paths are the same, assume in-place operation
        if video_path != output_path:
            # Copy video to output (simulated)
            try:
                with open(video_path, 'rb') as src, open(output_path, 'wb') as dst:
                    dst.write(src.read())
                logger.info("Subtitles added successfully (simulated)")
            except Exception as e:
                logger.error(f"Error adding subtitles: {e}")


def main():
    """Example usage of VideoTranslator."""
    # Create translator
    translator = VideoTranslator()
    
    # Example translation
    translator.translate_video(
        input_path="example/video.mp4",
        output_path="example/translated_video.mp4",
        source_lang="en",
        target_lang="es"
    )


if __name__ == "__main__":
    main() 