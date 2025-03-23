import pytest
import os
from pathlib import Path
from unittest.mock import Mock, patch
from ..models.config import AppConfig
from ..services.video_processor import VideoProcessor

@pytest.fixture
def config():
    """Create test configuration."""
    return AppConfig(
        storage=AppConfig.StorageConfig(
            storage_dir="test_storage",
            upload_dir="test_uploads",
            output_dir="test_outputs"
        )
    )

@pytest.fixture
def video_processor(config):
    """Create video processor instance."""
    return VideoProcessor(config)

@pytest.fixture
def test_video():
    """Create test video file."""
    video_path = Path("test_uploads/test_video.mp4")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.touch()
    return str(video_path)

@pytest.mark.asyncio
async def test_process_video_edit(video_processor, test_video):
    """Test video edit processing."""
    # Arrange
    output_path = str(Path("test_outputs/edited_test_video.mp4"))
    target_duration = 30.0
    target_width = 1080
    target_height = 1920
    target_lufs = -14.0

    # Mock ffmpeg.probe
    with patch('ffmpeg.probe') as mock_probe:
        mock_probe.return_value = {
            'streams': [
                {
                    'codec_type': 'video',
                    'width': '1920',
                    'height': '1080'
                }
            ],
            'format': {
                'duration': '60.0'
            }
        }

        # Mock ffmpeg input/output
        with patch('ffmpeg.input') as mock_input:
            mock_input.return_value = Mock()
            mock_input.return_value.filter.return_value = Mock()
            mock_input.return_value.filter.return_value.filter.return_value = Mock()
            mock_input.return_value.filter.return_value.filter.return_value.filter.return_value = Mock()
            mock_input.return_value.filter.return_value.filter.return_value.filter.return_value.output.return_value = Mock()
            mock_input.return_value.filter.return_value.filter.return_value.filter.return_value.output.return_value.run_async.return_value = Mock()
            mock_input.return_value.filter.return_value.filter.return_value.filter.return_value.output.return_value.run_async.return_value.communicate.return_value = (b'', b'')

            # Act
            output_path, duration = await video_processor.process_video_edit(
                input_path=test_video,
                output_path=output_path,
                target_duration=target_duration,
                target_width=target_width,
                target_height=target_height,
                target_lufs=target_lufs
            )

            # Assert
            assert output_path == str(Path("test_outputs/edited_test_video.mp4"))
            assert duration == 60.0
            mock_probe.assert_called_once_with(test_video)
            mock_input.assert_called_once_with(test_video)

@pytest.mark.asyncio
async def test_process_video_translate(video_processor, test_video):
    """Test video translation processing."""
    # Arrange
    output_path = str(Path("test_outputs/translated_test_video.mp4"))
    target_language = "es"
    voice_id = "voice_1"

    # Mock ffmpeg.probe
    with patch('ffmpeg.probe') as mock_probe:
        mock_probe.return_value = {
            'format': {
                'duration': '60.0'
            }
        }

        # Act
        output_path, duration = await video_processor.process_video_translate(
            input_path=test_video,
            output_path=output_path,
            target_language=target_language,
            voice_id=voice_id
        )

        # Assert
        assert output_path == str(Path("test_outputs/translated_test_video.mp4"))
        assert duration == 60.0
        mock_probe.assert_called_once_with(test_video)

@pytest.mark.asyncio
async def test_process_avatar_create(video_processor, test_video):
    """Test avatar creation processing."""
    # Arrange
    output_path = str(Path("test_outputs/avatar_test.glb"))
    avatar_type = "realistic"
    style = {"style": "casual"}

    # Mock ffmpeg.probe
    with patch('ffmpeg.probe') as mock_probe:
        mock_probe.return_value = {
            'format': {
                'duration': '60.0'
            }
        }

        # Act
        output_path, duration = await video_processor.process_avatar_create(
            input_path=test_video,
            output_path=output_path,
            avatar_type=avatar_type,
            style=style
        )

        # Assert
        assert output_path == str(Path("test_outputs/avatar_test.glb"))
        assert duration == 60.0
        mock_probe.assert_called_once_with(test_video)

@pytest.mark.asyncio
async def test_process_avatar_generate(video_processor):
    """Test avatar generation processing."""
    # Arrange
    avatar_id = "avatar_1"
    output_path = str(Path("test_outputs/generated_avatar_1.mp4"))
    script = "Hello, world!"
    voice_id = "voice_1"

    # Mock ffmpeg input/output
    with patch('ffmpeg.input') as mock_input:
        mock_input.return_value = Mock()
        mock_input.return_value.output.return_value = Mock()
        mock_input.return_value.output.return_value.run_async.return_value = Mock()
        mock_input.return_value.output.return_value.run_async.return_value.communicate.return_value = (b'', b'')

        # Act
        output_path, duration = await video_processor.process_avatar_generate(
            avatar_id=avatar_id,
            output_path=output_path,
            script=script,
            voice_id=voice_id
        )

        # Assert
        assert output_path == str(Path("test_outputs/generated_avatar_1.mp4"))
        assert duration == 5.0
        mock_input.assert_called_once()

@pytest.mark.asyncio
async def test_process_video_edit_error(video_processor, test_video):
    """Test video edit error handling."""
    # Arrange
    output_path = str(Path("test_outputs/edited_test_video.mp4"))
    target_duration = 30.0
    target_width = 1080
    target_height = 1920
    target_lufs = -14.0

    # Mock ffmpeg.probe to raise an exception
    with patch('ffmpeg.probe', side_effect=Exception("FFmpeg error")):
        # Act & Assert
        with pytest.raises(Exception):
            await video_processor.process_video_edit(
                input_path=test_video,
                output_path=output_path,
                target_duration=target_duration,
                target_width=target_width,
                target_height=target_height,
                target_lufs=target_lufs
            ) 