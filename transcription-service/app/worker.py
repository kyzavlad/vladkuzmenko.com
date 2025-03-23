import os
import sys
import json
import logging
import asyncio
import uuid
import time
from datetime import datetime
from pathlib import Path
import signal
import traceback

# Add parent directory to path to allow importing app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.transcription import Transcription, TranscriptionJob
from app.services.queue import start_consuming
from app.services.whisper import transcribe_audio, extract_audio_from_video, format_transcription_as_srt, format_transcription_as_vtt
from app.services.auth import notify_video_processing_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("transcription_worker")

# Flag to track if we're shutting down
is_shutting_down = False

async def process_job(message):
    """
    Process a transcription job message from the queue
    
    Args:
        message: The message from the queue containing job information
    """
    job_id = message.get("job_id")
    transcription_id = message.get("transcription_id")
    video_id = message.get("video_id")
    media_url = message.get("media_url")
    worker_id = message.get("worker_id")
    parameters = message.get("parameters", {})
    
    if not all([job_id, transcription_id, media_url]):
        logger.error(f"Invalid message format: {message}")
        return
    
    logger.info(f"Processing job {job_id} for transcription {transcription_id}")
    
    # Database session
    db = SessionLocal()
    temp_files = []
    
    try:
        # Update job status
        job = db.query(TranscriptionJob).filter(TranscriptionJob.id == job_id).first()
        if not job:
            logger.error(f"Job {job_id} not found in database")
            return
            
        job.status = "in_progress"
        job.started_at = datetime.utcnow()
        job.worker_id = worker_id
        
        # Update transcription status
        transcription = db.query(Transcription).filter(Transcription.id == transcription_id).first()
        if not transcription:
            logger.error(f"Transcription {transcription_id} not found in database")
            job.status = "failed"
            job.error = "Transcription not found in database"
            db.commit()
            return
            
        transcription.status = "processing"
        transcription.started_at = datetime.utcnow()
        transcription.progress = 0.0
        db.commit()
        
        # Extract audio if needed
        is_video = transcription.media_format.lower() not in ["mp3", "wav", "m4a", "mpga"]
        audio_path = None
        
        if is_video:
            logger.info(f"Extracting audio from video: {media_url}")
            transcription.progress = 0.1
            db.commit()
            
            try:
                # Extract audio from video
                audio_path = extract_audio_from_video(media_url)
                temp_files.append(audio_path)
                logger.info(f"Audio extracted to: {audio_path}")
            except Exception as e:
                logger.error(f"Failed to extract audio: {str(e)}")
                raise
        else:
            audio_path = media_url
        
        # Update progress
        transcription.progress = 0.2
        db.commit()
        
        # Get transcription parameters
        language = parameters.get("language", settings.DEFAULT_LANGUAGE)
        word_timestamps = parameters.get("word_timestamps", settings.WORD_LEVEL_TIMESTAMPS)
        prompt = parameters.get("prompt")
        
        # Perform transcription
        logger.info(f"Starting transcription of audio: {audio_path}")
        transcription_result = await transcribe_audio(
            audio_file_path=audio_path,
            language=language,
            word_timestamps=word_timestamps,
            prompt=prompt
        )
        
        # Update progress
        transcription.progress = 0.8
        db.commit()
        
        # Extract results
        full_text = transcription_result.get("text", "")
        segments = transcription_result.get("segments", [])
        words = transcription_result.get("words", [])
        
        # Create storage directory if using local storage
        storage_dir = Path(settings.LOCAL_STORAGE_PATH)
        os.makedirs(storage_dir, exist_ok=True)
        
        # Generate unique identifier for output files
        file_id = uuid.uuid4().hex
        
        # Save output files
        output_files = {}
        
        # Save JSON file
        json_path = os.path.join(storage_dir, f"{file_id}.json")
        with open(json_path, "w") as f:
            json.dump(transcription_result, f, indent=2)
        output_files["json"] = json_path
        
        # Save SRT file if segments exist
        if segments:
            srt_content = format_transcription_as_srt(segments)
            srt_path = os.path.join(storage_dir, f"{file_id}.srt")
            with open(srt_path, "w") as f:
                f.write(srt_content)
            output_files["srt"] = srt_path
        
        # Save VTT file if segments exist
        if segments:
            vtt_content = format_transcription_as_vtt(segments)
            vtt_path = os.path.join(storage_dir, f"{file_id}.vtt")
            with open(vtt_path, "w") as f:
                f.write(vtt_content)
            output_files["vtt"] = vtt_path
        
        # Save plain text file
        txt_path = os.path.join(storage_dir, f"{file_id}.txt")
        with open(txt_path, "w") as f:
            f.write(full_text)
        output_files["txt"] = txt_path
        
        # Update transcription in database
        transcription.status = "completed"
        transcription.progress = 1.0
        transcription.full_text = full_text
        transcription.segments = segments
        transcription.words = words if word_timestamps else None
        transcription.json_path = output_files.get("json")
        transcription.srt_path = output_files.get("srt")
        transcription.vtt_path = output_files.get("vtt")
        transcription.txt_path = output_files.get("txt")
        transcription.completed_at = datetime.utcnow()
        
        # Calculate processing duration
        if transcription.started_at:
            start_time = transcription.started_at
            end_time = transcription.completed_at or datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            
            # Update job with processing time
            job.processing_time = processing_time
        
        # Mark job as completed
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        job.result = {
            "output_files": output_files,
            "duration": transcription_result.get("duration"),
            "language": transcription_result.get("language"),
            "words_count": len(words) if words else 0,
            "segments_count": len(segments) if segments else 0,
        }
        
        db.commit()
        logger.info(f"Transcription {transcription_id} completed successfully")
        
        # Notify Video Processing Service
        try:
            file_paths = {
                "json": transcription.json_path,
                "srt": transcription.srt_path,
                "vtt": transcription.vtt_path,
                "txt": transcription.txt_path,
            }
            notify_video_processing_service(
                video_id=video_id,
                transcription_id=transcription_id,
                status="completed",
                files=file_paths
            )
        except Exception as e:
            logger.error(f"Failed to notify Video Processing Service: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Update job status
        try:
            job.status = "failed"
            job.error = str(e)
            job.completed_at = datetime.utcnow()
            
            if transcription:
                transcription.status = "failed"
                transcription.error = str(e)
                transcription.completed_at = datetime.utcnow()
                
            db.commit()
            
            # Notify Video Processing Service about failure
            try:
                notify_video_processing_service(
                    video_id=video_id,
                    transcription_id=transcription_id,
                    status="failed"
                )
            except Exception as notify_error:
                logger.error(f"Failed to notify Video Processing Service about failure: {str(notify_error)}")
                
        except Exception as db_error:
            logger.error(f"Error updating job status: {str(db_error)}")
    
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception:
                pass
        
        # Close database session
        db.close()

def handle_message(message):
    """
    Handle a message from the queue
    
    This function is passed to the queue consumer and called for each message.
    It runs the async process_job function in an event loop.
    
    Args:
        message: The message from the queue
    """
    if is_shutting_down:
        logger.warning("Worker is shutting down, not processing new messages")
        return
        
    # Create an event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Run the async process_job function
        loop.run_until_complete(process_job(message))
    finally:
        # Close the event loop
        loop.close()

def handle_shutdown(signum, frame):
    """
    Handle shutdown signals (SIGINT, SIGTERM)
    
    This function is called when the worker receives a shutdown signal.
    It sets the is_shutting_down flag to prevent processing new messages.
    """
    global is_shutting_down
    logger.info(f"Received shutdown signal {signum}, stopping worker...")
    is_shutting_down = True
    sys.exit(0)

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    # Generate a unique worker ID
    worker_id = f"worker-{uuid.uuid4().hex[:8]}"
    logger.info(f"Starting transcription worker {worker_id}")
    
    try:
        # Start consuming messages from the queue
        start_consuming(worker_id, handle_message)
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
    except Exception as e:
        logger.error(f"Worker crashed: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1) 