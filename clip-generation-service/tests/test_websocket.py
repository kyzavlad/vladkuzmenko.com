import pytest
from unittest.mock import Mock, patch
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime
from ..models.database import Job, JobStatus, JobType
from ..models.websocket import WebSocketMessage, WebSocketMessageType
from ..services.websocket import WebSocketManager

@pytest.fixture
def websocket():
    """Create mock WebSocket."""
    return Mock(spec=WebSocket)

@pytest.fixture
def websocket_manager():
    """Create WebSocket manager instance."""
    return WebSocketManager()

@pytest.fixture
def test_job():
    """Create a test job."""
    return Job(
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

@pytest.mark.asyncio
async def test_connect(websocket_manager, websocket):
    """Test WebSocket connection."""
    # Act
    await websocket_manager.connect(websocket, "test_user_1")

    # Assert
    websocket.accept.assert_called_once()
    assert "test_user_1" in websocket_manager.active_connections
    assert websocket in websocket_manager.active_connections["test_user_1"]

@pytest.mark.asyncio
async def test_disconnect(websocket_manager, websocket):
    """Test WebSocket disconnection."""
    # Arrange
    websocket_manager.active_connections = {
        "test_user_1": {websocket}
    }

    # Act
    await websocket_manager.disconnect(websocket, "test_user_1")

    # Assert
    assert "test_user_1" not in websocket_manager.active_connections

@pytest.mark.asyncio
async def test_broadcast_job_update(websocket_manager, websocket, test_job):
    """Test broadcasting job update."""
    # Arrange
    websocket_manager.active_connections = {
        "test_user_1": {websocket}
    }

    # Act
    await websocket_manager.broadcast_job_update(test_job)

    # Assert
    websocket.send_json.assert_called_once()
    message = websocket.send_json.call_args[0][0]
    assert message["type"] == WebSocketMessageType.JOB_UPDATE
    assert message["data"]["job_id"] == test_job.job_id
    assert message["data"]["status"] == test_job.status
    assert message["data"]["progress"] == test_job.progress

@pytest.mark.asyncio
async def test_broadcast_job_completion(websocket_manager, websocket, test_job):
    """Test broadcasting job completion."""
    # Arrange
    websocket_manager.active_connections = {
        "test_user_1": {websocket}
    }
    test_job.status = JobStatus.COMPLETED
    test_job.completed_at = datetime.utcnow()
    test_job.output_data = {"output_path": "output.mp4"}

    # Act
    await websocket_manager.broadcast_job_completion(test_job)

    # Assert
    websocket.send_json.assert_called_once()
    message = websocket.send_json.call_args[0][0]
    assert message["type"] == WebSocketMessageType.JOB_COMPLETED
    assert message["data"]["job_id"] == test_job.job_id
    assert message["data"]["output_data"] == test_job.output_data

@pytest.mark.asyncio
async def test_broadcast_job_error(websocket_manager, websocket, test_job):
    """Test broadcasting job error."""
    # Arrange
    websocket_manager.active_connections = {
        "test_user_1": {websocket}
    }
    error = "Processing failed"

    # Act
    await websocket_manager.broadcast_job_error(test_job, error)

    # Assert
    websocket.send_json.assert_called_once()
    message = websocket.send_json.call_args[0][0]
    assert message["type"] == WebSocketMessageType.JOB_ERROR
    assert message["data"]["job_id"] == test_job.job_id
    assert message["data"]["error"] == error

@pytest.mark.asyncio
async def test_handle_client(websocket_manager, websocket):
    """Test handling WebSocket client."""
    # Arrange
    websocket.receive_json.return_value = {
        "type": WebSocketMessageType.PING,
        "data": {}
    }

    # Act
    await websocket_manager.handle_client(websocket, "test_user_1")

    # Assert
    websocket.accept.assert_called_once()
    websocket.send_json.assert_called_once()
    message = websocket.send_json.call_args[0][0]
    assert message["type"] == WebSocketMessageType.PONG

@pytest.mark.asyncio
async def test_handle_client_disconnect(websocket_manager, websocket):
    """Test handling WebSocket client disconnection."""
    # Arrange
    websocket.receive_json.side_effect = WebSocketDisconnect()

    # Act
    await websocket_manager.handle_client(websocket, "test_user_1")

    # Assert
    websocket.accept.assert_called_once()
    assert "test_user_1" not in websocket_manager.active_connections

@pytest.mark.asyncio
async def test_handle_client_error(websocket_manager, websocket):
    """Test handling WebSocket client error."""
    # Arrange
    websocket.receive_json.side_effect = Exception("Connection error")

    # Act
    await websocket_manager.handle_client(websocket, "test_user_1")

    # Assert
    websocket.accept.assert_called_once()
    assert "test_user_1" not in websocket_manager.active_connections

@pytest.mark.asyncio
async def test_broadcast_to_disconnected_client(websocket_manager, websocket, test_job):
    """Test broadcasting to disconnected client."""
    # Arrange
    websocket_manager.active_connections = {
        "test_user_1": {websocket}
    }
    websocket.send_json.side_effect = WebSocketDisconnect()

    # Act
    await websocket_manager.broadcast_job_update(test_job)

    # Assert
    assert "test_user_1" not in websocket_manager.active_connections 