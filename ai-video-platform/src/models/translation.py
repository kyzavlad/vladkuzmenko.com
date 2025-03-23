from sqlalchemy import Column, Integer, String, Float, ForeignKey, JSON, Boolean
from sqlalchemy.orm import relationship

from .base import Base

class Translation(Base):
    job_id = Column(Integer, ForeignKey("job.id"), nullable=False)
    
    # Language settings
    source_language = Column(String, nullable=False)
    target_language = Column(String, nullable=False)
    preserve_voice_tone = Column(Boolean, default=True)
    
    # Translation details
    status = Column(String, nullable=False)
    progress = Column(Float, default=0.0)
    
    # Input/Output
    input_url = Column(String, nullable=False)
    output_url = Column(String)
    subtitles_url = Column(String)
    
    # Quality metrics
    translation_accuracy = Column(Float)
    lip_sync_quality = Column(Float)
    voice_preservation = Column(Float)
    
    # Processing details
    processing_time = Column(Float)  # in seconds
    token_cost = Column(Float)
    
    # Additional data
    metadata = Column(JSON)
    error_message = Column(String)
    
    # Relationships
    job = relationship("Job")
    segments = relationship("TranslationSegment", back_populates="translation")

class TranslationSegment(Base):
    translation_id = Column(Integer, ForeignKey("translation.id"), nullable=False)
    
    # Segment details
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    source_text = Column(String, nullable=False)
    translated_text = Column(String, nullable=False)
    
    # Quality metrics
    confidence_score = Column(Float)
    lip_sync_quality = Column(Float)
    
    # Additional data
    metadata = Column(JSON)
    
    # Relationships
    translation = relationship("Translation", back_populates="segments") 