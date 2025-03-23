from typing import Dict, Set, Optional
from fastapi import WebSocket, WebSocketDisconnect
import json
import logging
from datetime import datetime
from ..models.database import Job, JobStatus
from ..models.websocket import WebSocketMessage, WebSocketMessageType

logger = logging.getLogger(__name__)

class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self._running = False

    async def connect(self, websocket: WebSocket, user_id: str):
        """Connect a new WebSocket client."""
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        self.active_connections[user_id].add(websocket)
        logger.info(f"WebSocket client connected for user {user_id}")

    async def disconnect(self, websocket: WebSocket, user_id: str):
        """Disconnect a WebSocket client."""
        if user_id in self.active_connections:
            self.active_connections[user_id].remove(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
        logger.info(f"WebSocket client disconnected for user {user_id}")

    async def broadcast_job_update(self, job: Job):
        """Broadcast job update to all connected clients for the job's user."""
        if job.user_id not in self.active_connections:
            return

        message = WebSocketMessage(
            type=WebSocketMessageType.JOB_UPDATE,
            data={
                "job_id": job.job_id,
                "status": job.status,
                "progress": job.progress,
                "error_message": job.error_message,
                "updated_at": job.updated_at.isoformat()
            }
        )

        await self._broadcast_to_user(job.user_id, message)

    async def broadcast_job_completion(self, job: Job):
        """Broadcast job completion to all connected clients for the job's user."""
        if job.user_id not in self.active_connections:
            return

        message = WebSocketMessage(
            type=WebSocketMessageType.JOB_COMPLETED,
            data={
                "job_id": job.job_id,
                "output_data": job.output_data,
                "completed_at": job.completed_at.isoformat()
            }
        )

        await self._broadcast_to_user(job.user_id, message)

    async def broadcast_job_error(self, job: Job, error: str):
        """Broadcast job error to all connected clients for the job's user."""
        if job.user_id not in self.active_connections:
            return

        message = WebSocketMessage(
            type=WebSocketMessageType.JOB_ERROR,
            data={
                "job_id": job.job_id,
                "error": error,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

        await self._broadcast_to_user(job.user_id, message)

    async def _broadcast_to_user(self, user_id: str, message: WebSocketMessage):
        """Broadcast message to all connected clients for a user."""
        if user_id not in self.active_connections:
            return

        disconnected = set()
        for connection in self.active_connections[user_id]:
            try:
                await connection.send_json(message.dict())
            except WebSocketDisconnect:
                disconnected.add(connection)
            except Exception as e:
                logger.error(f"Error sending WebSocket message: {str(e)}")
                disconnected.add(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            await self.disconnect(connection, user_id)

    async def handle_client(self, websocket: WebSocket, user_id: str):
        """Handle WebSocket client connection."""
        try:
            await self.connect(websocket, user_id)
            while True:
                try:
                    data = await websocket.receive_json()
                    message = WebSocketMessage(**data)
                    
                    # Handle different message types
                    if message.type == WebSocketMessageType.PING:
                        await websocket.send_json({
                            "type": WebSocketMessageType.PONG,
                            "timestamp": datetime.utcnow().isoformat()
                        })
                    else:
                        logger.warning(f"Unhandled message type: {message.type}")

                except WebSocketDisconnect:
                    break
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received from WebSocket client")
                except Exception as e:
                    logger.error(f"Error handling WebSocket message: {str(e)}")

        except Exception as e:
            logger.error(f"Error in WebSocket handler: {str(e)}")
        finally:
            await self.disconnect(websocket, user_id) 