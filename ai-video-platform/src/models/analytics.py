from sqlalchemy import Column, Integer, String, Float, ForeignKey, JSON, DateTime
from sqlalchemy.orm import relationship

from .base import Base

class JobAnalytics(Base):
    job_id = Column(Integer, ForeignKey("job.id"), nullable=False)
    
    # Performance metrics
    processing_duration = Column(Float)  # in seconds
    gpu_utilization = Column(Float)
    memory_usage = Column(Float)
    cpu_utilization = Column(Float)
    
    # Resource usage
    storage_used = Column(Float)  # in MB
    network_bandwidth = Column(Float)  # in MB/s
    cache_hit_rate = Column(Float)
    
    # Quality metrics
    output_size = Column(Float)  # in MB
    compression_ratio = Column(Float)
    quality_score = Column(Float)
    
    # System metrics
    queue_wait_time = Column(Float)  # in seconds
    processing_start_time = Column(DateTime)
    processing_end_time = Column(DateTime)
    
    # Detailed metrics
    performance_data = Column(JSON)
    resource_usage_data = Column(JSON)
    quality_data = Column(JSON)
    
    # Relationships
    job = relationship("Job", back_populates="analytics") 