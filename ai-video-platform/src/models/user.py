from sqlalchemy import Boolean, Column, Integer, String, Float, ForeignKey, Enum
from sqlalchemy.orm import relationship
import enum

from .base import Base

class UserRole(str, enum.Enum):
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"

class User(Base):
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    role = Column(Enum(UserRole), default=UserRole.USER)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    token_balance = Column(Float, default=0.0)
    
    # Relationships
    jobs = relationship("Job", back_populates="user")
    avatars = relationship("Avatar", back_populates="user")
    subscriptions = relationship("Subscription", back_populates="user")
    billing_history = relationship("BillingHistory", back_populates="user") 