import pytest
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch
from ..models.database import Job, JobStatus, JobType
from ..models.config import AppConfig
from ..services.job_processor import JobProcessor
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
def job_repository():
    """Create mock job repository."""
    return Mock()

@pytest.fixture
def job_log_repository():
    """Create mock job log repository."""
    return Mock()

@pytest.fixture
def video_processor():
    """Create mock video processor."""
    return Mock(spec=VideoProcessor)

@pytest.fixture
def job_processor(config, job_repository, job_log_repository, video_processor):
    """Create job processor with mocked dependencies."""
    with patch('..services.job_processor.VideoProcessor', return_value=video_processor):
        processor = JobProcessor(job_repository, job_log_repository, config)
        return processor

@pytest.fixture
def test_job():
    """Create a test job."""
    return Job(
        id="1",
        job_id="test_job_1",
        user_id="test_user_1",
        job_type=JobType.VIDEO_EDIT,
        status=JobStatus.PENDING,
        progress=0.0,
        input_data={
            "input_path": "test_video.mp4",
            "target_duration": 30.0,
            "target_width": 1080,
            "target_height": 1920,
            "target_lufs": -14.0
        },
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

@pytest.mark.asyncio
async def test_create_job(job_processor, job_repository, job_log_repository):
    """Test job creation."""
    # Arrange
    job_data = {
        "job_id": "test_job_1",
        "user_id": "test_user_1",
        "job_type": JobType.VIDEO_EDIT,
        "input_data": {"input_path": "test_video.mp4"}
    }
    job_repository.create.return_value = Job(**job_data)

    # Act
    job = await job_processor.create_job(job_data)

    # Assert
    assert job.job_id == "test_job_1"
    assert job.user_id == "test_user_1"
    assert job.job_type == JobType.VIDEO_EDIT
    assert job.status == JobStatus.PENDING
    assert job.progress == 0.0
    job_repository.create.assert_called_once()
    job_log_repository.create.assert_called_once()

@pytest.mark.asyncio
async def test_process_video_edit(job_processor, job_repository, job_log_repository, video_processor, test_job):
    """Test video edit job processing."""
    # Arrange
    job_repository.get_by_id.return_value = test_job
    job_repository.update_status.return_value = test_job
    job_repository.update.return_value = test_job
    video_processor.process_video_edit.return_value = ("output.mp4", 30.0)

    # Act
    job = await job_processor.process_job("test_job_1")

    # Assert
    assert job.status == JobStatus.COMPLETED
    assert job.progress == 100.0
    job_repository.get_by_id.assert_called_once_with("test_job_1")
    job_repository.update_status.assert_called()
    video_processor.process_video_edit.assert_called_once()
    job_repository.update.assert_called_once()
    job_log_repository.create.assert_called()

@pytest.mark.asyncio
async def test_process_job_error(job_processor, job_repository, job_log_repository, test_job):
    """Test job processing error handling."""
    # Arrange
    job_repository.get_by_id.return_value = test_job
    job_repository.update_status.return_value = test_job
    job_repository.update.side_effect = Exception("Processing error")

    # Act & Assert
    with pytest.raises(Exception):
        await job_processor.process_job("test_job_1")

    # Verify error handling
    job_repository.update_status.assert_called_with(
        job_id=test_job.job_id,
        status=JobStatus.FAILED,
        error_message="Processing error"
    )
    job_log_repository.create.assert_called()

@pytest.mark.asyncio
async def test_update_job_progress(job_processor, job_repository, job_log_repository):
    """Test job progress update."""
    # Arrange
    job_repository.update_status.return_value = Job(
        id="1",
        job_id="test_job_1",
        user_id="test_user_1",
        job_type=JobType.VIDEO_EDIT,
        status=JobStatus.PROCESSING,
        progress=50.0,
        input_data={},
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

    # Act
    job = await job_processor.update_job_progress("test_job_1", 50.0)

    # Assert
    assert job.progress == 50.0
    job_repository.update_status.assert_called_once_with(
        job_id="test_job_1",
        progress=50.0
    )
    job_log_repository.create.assert_called_once()

@pytest.mark.asyncio
async def test_cancel_job(job_processor, job_repository, job_log_repository):
    """Test job cancellation."""
    # Arrange
    job_repository.update_status.return_value = Job(
        id="1",
        job_id="test_job_1",
        user_id="test_user_1",
        job_type=JobType.VIDEO_EDIT,
        status=JobStatus.CANCELLED,
        progress=0.0,
        input_data={},
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

    # Act
    job = await job_processor.cancel_job("test_job_1")

    # Assert
    assert job.status == JobStatus.CANCELLED
    job_repository.update_status.assert_called_once_with(
        job_id="test_job_1",
        status=JobStatus.CANCELLED
    )
    job_log_repository.create.assert_called_once() 