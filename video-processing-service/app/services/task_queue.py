import json
import pika
from typing import Dict, Any, Optional
import logging
from datetime import datetime
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.video import ProcessingTask

# Configure logging
logger = logging.getLogger(__name__)

class QueueException(Exception):
    """Exception raised for queue-related errors"""
    pass

def get_rabbitmq_connection():
    """
    Establish a connection to the RabbitMQ server
    
    Returns:
        pika.BlockingConnection: A connection to the RabbitMQ server
        
    Raises:
        QueueException: If the connection cannot be established
    """
    try:
        credentials = pika.PlainCredentials(
            settings.RABBITMQ_USER,
            settings.RABBITMQ_PASSWORD
        )
        
        parameters = pika.ConnectionParameters(
            host=settings.RABBITMQ_HOST,
            port=settings.RABBITMQ_PORT,
            credentials=credentials,
            heartbeat=600,
            blocked_connection_timeout=300
        )
        
        return pika.BlockingConnection(parameters)
    except Exception as e:
        raise QueueException(f"Failed to connect to RabbitMQ: {str(e)}")

def queue_processing_task(task_id: str):
    """
    Queue a video processing task for asynchronous execution
    
    This function looks up the task in the database and adds it
    to the appropriate RabbitMQ queue based on its type.
    
    Args:
        task_id: The ID of the processing task to queue
        
    Raises:
        QueueException: If the task cannot be queued
    """
    # Get task from database
    db = SessionLocal()
    try:
        task = db.query(ProcessingTask).filter(ProcessingTask.id == task_id).first()
        if not task:
            raise QueueException(f"Task with ID {task_id} not found")
        
        # Determine queue name based on task type
        queue_name = f"video_processing_{task.task_type}"
        
        # Connect to RabbitMQ
        connection = get_rabbitmq_connection()
        channel = connection.channel()
        
        # Ensure queue exists (creates it if not)
        channel.queue_declare(queue=queue_name, durable=True)
        
        # Prepare message with task ID and parameters
        message = {
            "task_id": task.id,
            "video_id": task.video_id,
            "parameters": task.parameters
        }
        
        # Publish message to queue
        channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=2,  # make message persistent
                priority=task.priority
            )
        )
        
        # Update task status in database
        task.status = "queued"
        db.commit()
        
        # Close connection
        connection.close()
        
        logger.info(f"Task {task_id} queued successfully to {queue_name}")
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error queueing task {task_id}: {str(e)}")
        raise QueueException(f"Failed to queue task: {str(e)}")
    finally:
        db.close()

def update_task_status(task_id: str, status: str, result: Optional[Dict[str, Any]] = None, error_message: Optional[str] = None):
    """
    Update the status of a processing task in the database
    
    Args:
        task_id: The ID of the task to update
        status: The new status ('in_progress', 'completed', 'failed')
        result: Optional result data if the task was completed
        error_message: Optional error message if the task failed
        
    Raises:
        QueueException: If the task status cannot be updated
    """
    db = SessionLocal()
    try:
        task = db.query(ProcessingTask).filter(ProcessingTask.id == task_id).first()
        if not task:
            raise QueueException(f"Task with ID {task_id} not found")
        
        # Update task status and data
        task.status = status
        
        if status == "in_progress":
            task.started_at = datetime.utcnow()
        elif status in ["completed", "failed"]:
            task.completed_at = datetime.utcnow()
            
        if result is not None:
            task.result = result
            
        if error_message is not None:
            task.error_message = error_message
            
        db.commit()
        logger.info(f"Task {task_id} status updated to {status}")
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating task {task_id} status: {str(e)}")
        raise QueueException(f"Failed to update task status: {str(e)}")
    finally:
        db.close() 