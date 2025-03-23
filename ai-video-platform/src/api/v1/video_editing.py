from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
import asyncio
import time

from ...core.security import get_current_active_user
from ...core.optimization import ResourceOptimizer
from ...db.session import get_db
from ...models.user import User
from ...models.job import Job, JobStatus, JobType
from ...core.video import VideoProcessor
from ...core.audio import AudioProcessor
from ...core.ai import AIModelManager

router = APIRouter()
resource_optimizer = ResourceOptimizer()

@router.post("/process")
async def process_video(
    video_url: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Process a video with AI enhancements."""
    job_id = str(uuid.uuid4())
    
    # Create job record
    job = Job(
        user_id=current_user.id,
        type=JobType.VIDEO_EDIT,
        status=JobStatus.PENDING,
        input_url=video_url,
        parameters={}
    )
    db.add(job)
    db.commit()
    
    # Start processing in background
    background_tasks.add_task(
        process_video_background,
        job_id=job_id,
        video_url=video_url,
        user_id=current_user.id,
        db=db
    )
    
    return {"job_id": job_id, "status": "processing"}

async def process_video_background(
    job_id: str,
    video_url: str,
    user_id: int,
    db: Session
):
    """Background task for video processing."""
    try:
        # Update job status
        job = db.query(Job).filter(Job.id == job_id).first()
        job.status = JobStatus.PROCESSING
        db.commit()
        
        # Initialize processors
        video_processor = VideoProcessor()
        audio_processor = AudioProcessor()
        ai_manager = AIModelManager()
        
        # Load video
        frames, metadata = await resource_optimizer.optimize_processing(
            job_id=job_id,
            input_data=video_url,
            process_fn=video_processor.load_video,
            required_memory=metadata["frame_count"] * 3 * metadata["width"] * metadata["height"] * 4,  # Rough estimate
            optimization_config={"batch_size": 32}
        )
        
        # Process frames in batches
        processed_frames = await resource_optimizer.optimize_processing(
            job_id=job_id,
            input_data=frames,
            process_fn=video_processor.process_frames,
            required_memory=len(frames) * 3 * metadata["width"] * metadata["height"] * 4,
            optimization_config={
                "batch_size": 32,
                "quantization": True,
                "fuse": True
            }
        )
        
        # Extract and process audio
        audio_path = await resource_optimizer.optimize_processing(
            job_id=job_id,
            input_data=video_url,
            process_fn=video_processor.extract_audio,
            required_memory=1024 * 1024 * 100,  # 100MB estimate
            optimization_config={"batch_size": 1}
        )
        
        # Enhance audio
        waveform, sr = audio_processor.load_audio(audio_path)
        enhanced_audio = await resource_optimizer.optimize_processing(
            job_id=job_id,
            input_data=(waveform, sr),
            process_fn=audio_processor.enhance_audio,
            required_memory=len(waveform) * 4,  # 4 bytes per sample
            optimization_config={"batch_size": 1}
        )
        
        # Save enhanced audio
        enhanced_audio_path = f"enhanced_{audio_path}"
        audio_processor.save_audio(enhanced_audio, enhanced_audio_path, sr)
        
        # Save final video
        output_path = f"output_{job_id}.mp4"
        await resource_optimizer.optimize_processing(
            job_id=job_id,
            input_data=(processed_frames, enhanced_audio_path),
            process_fn=video_processor.save_video,
            required_memory=len(processed_frames) * 3 * metadata["width"] * metadata["height"] * 4,
            optimization_config={"batch_size": 1}
        )
        
        # Update job status
        job.status = JobStatus.COMPLETED
        job.output_url = output_path
        job.processing_time = time.time() - job.created_at.timestamp()
        db.commit()
        
    except Exception as e:
        logger.error(f"Error processing video: {e}", exc_info=True)
        job.status = JobStatus.FAILED
        job.error_message = str(e)
        db.commit()

@router.get("/status/{job_id}")
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get the status of a video processing job."""
    job = db.query(Job).filter(
        Job.id == job_id,
        Job.user_id == current_user.id
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job.id,
        "status": job.status,
        "progress": job.progress,
        "output_url": job.output_url,
        "error_message": job.error_message
    }

@router.get("/metrics")
async def get_processing_metrics():
    """Get current processing metrics."""
    return resource_optimizer.monitor_performance() 