from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class WebSocketMessageType(str, Enum):
    """WebSocket message types."""
    JOB_UPDATE = "job_update"
    JOB_COMPLETED = "job_completed"
    JOB_ERROR = "job_error"
    PING = "ping"
    PONG = "pong"

class WebSocketMessage(BaseModel):
    """Base WebSocket message model."""
    type: WebSocketMessageType
    data: Dict[str, Any]
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)

class JobUpdateMessage(WebSocketMessage):
    """Job update message."""
    type: WebSocketMessageType = WebSocketMessageType.JOB_UPDATE
    data: Dict[str, Any] = Field(
        ...,
        example={
            "job_id": "123",
            "status": "processing",
            "progress": 50.0,
            "error_message": None,
            "updated_at": "2024-03-21T10:00:00Z"
        }
    )

class JobCompletedMessage(WebSocketMessage):
    """Job completed message."""
    type: WebSocketMessageType = WebSocketMessageType.JOB_COMPLETED
    data: Dict[str, Any] = Field(
        ...,
        example={
            "job_id": "123",
            "output_data": {"url": "https://example.com/output.mp4"},
            "completed_at": "2024-03-21T10:00:00Z"
        }
    )

class JobErrorMessage(WebSocketMessage):
    """Job error message."""
    type: WebSocketMessageType = WebSocketMessageType.JOB_ERROR
    data: Dict[str, Any] = Field(
        ...,
        example={
            "job_id": "123",
            "error": "Failed to process video",
            "timestamp": "2024-03-21T10:00:00Z"
        }
    )

class PingMessage(WebSocketMessage):
    """Ping message."""
    type: WebSocketMessageType = WebSocketMessageType.PING
    data: Dict[str, Any] = Field(default_factory=dict)

class PongMessage(WebSocketMessage):
    """Pong message."""
    type: WebSocketMessageType = WebSocketMessageType.PONG
    data: Dict[str, Any] = Field(
        ...,
        example={
            "timestamp": "2024-03-21T10:00:00Z"
        }
    ) 