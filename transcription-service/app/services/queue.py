import json
import pika
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.transcription import TranscriptionJob, Transcription

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
        logger.error(f"Failed to connect to RabbitMQ: {str(e)}")
        raise QueueException(f"Failed to connect to RabbitMQ: {str(e)}")

def queue_transcription_job(job_id: str):
    """
    Queue a transcription job for asynchronous processing
    
    Args:
        job_id: The ID of the transcription job to queue
        
    Raises:
        QueueException: If the job cannot be queued
    """
    # Get job from database
    db = SessionLocal()
    try:
        job = db.query(TranscriptionJob).filter(TranscriptionJob.id == job_id).first()
        if not job:
            raise QueueException(f"Job with ID {job_id} not found")
        
        # Get associated transcription
        transcription = db.query(Transcription).filter(
            Transcription.id == job.transcription_id
        ).first()
        
        if not transcription:
            raise QueueException(f"Transcription with ID {job.transcription_id} not found")
        
        # Connect to RabbitMQ
        connection = get_rabbitmq_connection()
        channel = connection.channel()
        
        # Ensure queue exists (creates it if not)
        channel.queue_declare(queue=settings.RABBITMQ_QUEUE, durable=True)
        
        # Prepare message
        message = {
            "job_id": job.id,
            "transcription_id": job.transcription_id,
            "video_id": transcription.video_id,
            "media_url": transcription.media_url,
            "parameters": job.parameters,
            "priority": job.priority
        }
        
        # Publish message to queue
        channel.basic_publish(
            exchange='',
            routing_key=settings.RABBITMQ_QUEUE,
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=2,  # make message persistent
                priority=job.priority
            )
        )
        
        # Update job status
        job.status = "queued"
        db.commit()
        
        # Close connection
        connection.close()
        
        logger.info(f"Job {job_id} queued successfully")
    
    except Exception as e:
        db.rollback()
        logger.error(f"Error queueing job {job_id}: {str(e)}")
        raise QueueException(f"Failed to queue job: {str(e)}")
    
    finally:
        db.close()

def retry_failed_jobs():
    """
    Retry failed transcription jobs that have not exceeded their retry limit
    
    This function should be called periodically by a scheduler.
    """
    db = SessionLocal()
    try:
        # Find failed jobs eligible for retry
        now = datetime.utcnow()
        jobs = db.query(TranscriptionJob).filter(
            TranscriptionJob.status == "failed",
            TranscriptionJob.retry_count < TranscriptionJob.max_retries,
            (TranscriptionJob.next_retry_at.is_(None) | (TranscriptionJob.next_retry_at <= now))
        ).all()
        
        logger.info(f"Found {len(jobs)} failed jobs eligible for retry")
        
        for job in jobs:
            # Update retry information
            job.retry_count += 1
            
            # Exponential backoff for retries: 1min, 5min, 25min, etc.
            backoff_minutes = 5 ** (job.retry_count - 1)
            job.next_retry_at = now + timedelta(minutes=backoff_minutes)
            
            # Reset status to pending
            job.status = "pending"
            
            # Update transcription status
            transcription = db.query(Transcription).filter(
                Transcription.id == job.transcription_id
            ).first()
            
            if transcription:
                transcription.status = "pending"
                transcription.error = None
                
            db.commit()
            
            # Queue the job again
            try:
                queue_transcription_job(job.id)
                logger.info(f"Job {job.id} requeued for retry {job.retry_count}/{job.max_retries}")
            except Exception as e:
                logger.error(f"Failed to requeue job {job.id}: {str(e)}")
    
    except Exception as e:
        db.rollback()
        logger.error(f"Error retrying failed jobs: {str(e)}")
    
    finally:
        db.close()

def start_consuming(worker_id: str, callback_function):
    """
    Start consuming messages from the queue
    
    Args:
        worker_id: A unique identifier for this worker
        callback_function: The function to call when a message is received
        
    Returns:
        None
    """
    try:
        # Connect to RabbitMQ
        connection = get_rabbitmq_connection()
        channel = connection.channel()
        
        # Ensure queue exists
        channel.queue_declare(queue=settings.RABBITMQ_QUEUE, durable=True)
        
        # Set QoS prefetch to limit concurrent jobs per worker
        channel.basic_qos(prefetch_count=1)
        
        # Define callback wrapper that passes worker_id
        def callback_wrapper(ch, method, properties, body):
            try:
                # Parse message
                message = json.loads(body)
                
                # Add worker ID to message
                message["worker_id"] = worker_id
                
                # Call the actual callback function
                callback_function(message)
                
                # Acknowledge message (only if processing was successful)
                ch.basic_ack(delivery_tag=method.delivery_tag)
                
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                # Reject message and requeue
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
        
        # Start consuming
        channel.basic_consume(
            queue=settings.RABBITMQ_QUEUE,
            on_message_callback=callback_wrapper
        )
        
        logger.info(f"Worker {worker_id} started consuming from queue {settings.RABBITMQ_QUEUE}")
        
        # Start consuming (this blocks until channel.stop_consuming() is called)
        channel.start_consuming()
        
    except Exception as e:
        logger.error(f"Error in consumer {worker_id}: {str(e)}")
        raise 