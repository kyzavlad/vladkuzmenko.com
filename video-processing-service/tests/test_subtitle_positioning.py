import os
import sys
import unittest
import asyncio
from unittest.mock import MagicMock, patch
import tempfile
import json

# Add parent directory to path to import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.subtitles import (
    SubtitlePositioningService,
    ContentImportance,
    ContentRegion,
    TextPosition
)


class TestSubtitlePositioning(unittest.TestCase):
    """Tests for the SubtitlePositioningService class."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a mock video file
        self.video_path = os.path.join(self.temp_dir.name, "test_video.mp4")
        with open(self.video_path, "w") as f:
            f.write("mock video content")
        
        # Create a simple transcript
        self.transcript = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "This is the first subtitle"
                },
                {
                    "start": 2.5,
                    "end": 5.0,
                    "text": "This is the second subtitle"
                }
            ]
        }
        
        # Initialize the service with mock FFmpeg paths
        self.positioning_service = SubtitlePositioningService(
            ffmpeg_path="mock_ffmpeg",
            ffprobe_path="mock_ffprobe",
            config={
                "frame_sample_rate": 5,
                "position_preference": ["bottom", "top", "center"]
            }
        )
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    @patch("asyncio.create_subprocess_exec")
    @patch("cv2.imread")
    @patch("os.listdir")
    async def test_extract_key_frames(self, mock_listdir, mock_imread, mock_subprocess):
        """Test extracting key frames from a video."""
        # Mock FFmpeg process
        process_mock = MagicMock()
        process_mock.returncode = 0
        process_mock.communicate.return_value = asyncio.Future()
        process_mock.communicate.return_value.set_result((b"", b""))
        mock_subprocess.return_value = asyncio.Future()
        mock_subprocess.return_value.set_result(process_mock)
        
        # Mock frame files
        mock_listdir.return_value = ["frame_0001.jpg", "frame_0002.jpg"]
        
        # Mock video info
        video_info = {
            "duration": 10.0,
            "fps": 30,
            "width": 1920,
            "height": 1080
        }
        
        # Test the function
        frames = await self.positioning_service._extract_key_frames(
            self.video_path, video_info
        )
        
        # Verify the results
        self.assertEqual(len(frames), 2)
        self.assertTrue(all(isinstance(f[0], str) and isinstance(f[1], float) for f in frames))
    
    def test_analyze_frame(self):
        """Test analyzing a frame for content regions."""
        # Mock CV2 functionality
        with patch("cv2.imread") as mock_imread, \
             patch("cv2.cvtColor") as mock_cvtColor, \
             patch("cv2.Canny") as mock_canny:
            
            # Create a mock frame
            mock_frame = MagicMock()
            mock_frame.shape = (1080, 1920, 3)
            mock_imread.return_value = mock_frame
            
            # Configure other mocks
            mock_canny.return_value = MagicMock()
            
            # Disable face and object detection for this test
            self.positioning_service.enable_face_detection = False
            self.positioning_service.enable_object_detection = False
            
            # Create a simple video info dict
            video_info = {
                "width": 1920,
                "height": 1080
            }
            
            # Test with a mock frame path
            frame_path = os.path.join(self.temp_dir.name, "test_frame.jpg")
            with open(frame_path, "w") as f:
                f.write("mock frame content")
            
            regions = self.positioning_service._analyze_frame(frame_path, video_info)
            
            # Just verify it runs and returns a list without errors
            self.assertIsInstance(regions, list)
    
    def test_determine_optimal_position(self):
        """Test determining optimal subtitle position based on content analysis."""
        # Create test regions
        face_region = ContentRegion(
            x=100, y=100, width=100, height=100,
            importance=ContentImportance.CRITICAL,
            confidence=0.9,
            content_type="face"
        )
        
        text_region = ContentRegion(
            x=100, y=800, width=300, height=50,
            importance=ContentImportance.MEDIUM,
            confidence=0.8,
            content_type="text"
        )
        
        # Video info
        video_info = {
            "width": 1920,
            "height": 1080
        }
        
        # Test scenario 1: Face in center, text at bottom
        frame_analyses = [[face_region, text_region]]
        position = self.positioning_service._determine_optimal_position(frame_analyses, video_info)
        
        # Should recommend top position as face is in center and text is at bottom
        self.assertEqual(position, "top")
        
        # Test scenario 2: Nothing important in any region
        frame_analyses = [[]]
        position = self.positioning_service._determine_optimal_position(frame_analyses, video_info)
        
        # Should use first preference from config (bottom)
        self.assertEqual(position, "bottom")
    
    @patch.object(SubtitlePositioningService, "_get_video_info")
    @patch.object(SubtitlePositioningService, "_extract_key_frames")
    @patch.object(SubtitlePositioningService, "_analyze_frame")
    @patch.object(SubtitlePositioningService, "_determine_optimal_position")
    async def test_analyze_video_for_positioning(
        self, mock_determine_position, mock_analyze_frame, 
        mock_extract_frames, mock_get_video_info
    ):
        """Test analyzing a video for optimal subtitle positioning."""
        # Configure mocks
        mock_get_video_info.return_value = asyncio.Future()
        mock_get_video_info.return_value.set_result({
            "duration": 10.0,
            "fps": 30,
            "width": 1920,
            "height": 1080
        })
        
        mock_extract_frames.return_value = asyncio.Future()
        mock_extract_frames.return_value.set_result([
            (os.path.join(self.temp_dir.name, "frame_0001.jpg"), 1.0),
            (os.path.join(self.temp_dir.name, "frame_0002.jpg"), 4.0)
        ])
        
        mock_analyze_frame.return_value = []
        mock_determine_position.return_value = "top"
        
        # Run the analysis
        result = await self.positioning_service.analyze_video_for_positioning(
            self.video_path, self.transcript
        )
        
        # Verify the results
        self.assertEqual(len(result["segments"]), 2)
        self.assertEqual(result["segments"][0]["style"]["position"], "top")
        self.assertEqual(result["segments"][1]["style"]["position"], "top")
        
        # Verify the correct methods were called
        mock_get_video_info.assert_called_once()
        mock_extract_frames.assert_called_once()
        self.assertEqual(mock_analyze_frame.call_count, 2)
        self.assertEqual(mock_determine_position.call_count, 2)


if __name__ == "__main__":
    unittest.main() 