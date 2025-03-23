from sqlalchemy import Column, Integer, String, Float, ForeignKey, JSON
from sqlalchemy.orm import relationship

from .base import Base

class QualityMetrics(Base):
    job_id = Column(Integer, ForeignKey("job.id"), nullable=False)
    
    # Video metrics
    resolution = Column(String)
    frame_rate = Column(Float)
    bitrate = Column(Float)
    color_accuracy = Column(Float)
    motion_smoothness = Column(Float)
    artifact_level = Column(Float)
    
    # Audio metrics
    loudness = Column(Float)
    dynamic_range = Column(Float)
    frequency_response = Column(Float)
    distortion = Column(Float)
    noise_level = Column(Float)
    
    # Overall metrics
    overall_score = Column(Float)
    processing_quality = Column(Float)
    compression_quality = Column(Float)
    
    # Detailed analysis
    scene_analysis = Column(JSON)
    audio_analysis = Column(JSON)
    technical_analysis = Column(JSON)
    
    # Relationships
    job = relationship("Job", back_populates="quality_metrics") 