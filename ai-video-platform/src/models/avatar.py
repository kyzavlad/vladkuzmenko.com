from sqlalchemy import Column, Integer, String, Float, ForeignKey, JSON, Boolean
from sqlalchemy.orm import relationship

from .base import Base

class Avatar(Base):
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    
    # Avatar details
    name = Column(String, nullable=False)
    style = Column(String, nullable=False)
    status = Column(String, nullable=False)
    
    # Source materials
    video_sample_url = Column(String)
    additional_photos_urls = Column(JSON)
    voice_sample_url = Column(String)
    
    # Generated assets
    preview_url = Column(String)
    model_url = Column(String)
    voice_model_url = Column(String)
    
    # Configuration
    settings = Column(JSON)
    is_active = Column(Boolean, default=True)
    
    # Quality metrics
    face_quality_score = Column(Float)
    voice_quality_score = Column(Float)
    animation_quality_score = Column(Float)
    
    # Relationships
    user = relationship("User", back_populates="avatars")
    generations = relationship("AvatarGeneration", back_populates="avatar")

class AvatarGeneration(Base):
    avatar_id = Column(Integer, ForeignKey("avatar.id"), nullable=False)
    job_id = Column(Integer, ForeignKey("job.id"), nullable=False)
    
    # Generation details
    script_text = Column(String)
    emotion_markers = Column(JSON)
    background_type = Column(String)
    custom_background_url = Column(String)
    
    # Output
    output_url = Column(String)
    preview_url = Column(String)
    status = Column(String)
    
    # Quality metrics
    lip_sync_quality = Column(Float)
    emotion_accuracy = Column(Float)
    animation_smoothness = Column(Float)
    
    # Relationships
    avatar = relationship("Avatar", back_populates="generations")
    job = relationship("Job") 