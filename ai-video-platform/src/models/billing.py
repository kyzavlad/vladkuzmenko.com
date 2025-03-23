from sqlalchemy import Column, Integer, String, Float, ForeignKey, JSON, Enum, DateTime, Boolean
from sqlalchemy.orm import relationship
import enum

from .base import Base

class SubscriptionPlan(str, enum.Enum):
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"

class PaymentStatus(str, enum.Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"

class Subscription(Base):
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    plan = Column(Enum(SubscriptionPlan), nullable=False)
    
    # Subscription details
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime)
    auto_renew = Column(Boolean, default=True)
    is_active = Column(Boolean, default=True)
    
    # Payment details
    payment_method_id = Column(String)
    last_payment_date = Column(DateTime)
    next_payment_date = Column(DateTime)
    
    # Features and limits
    features = Column(JSON)
    usage_limits = Column(JSON)
    
    # Relationships
    user = relationship("User", back_populates="subscriptions")
    payments = relationship("Payment", back_populates="subscription")

class Payment(Base):
    subscription_id = Column(Integer, ForeignKey("subscription.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    
    # Payment details
    amount = Column(Float, nullable=False)
    currency = Column(String, nullable=False)
    status = Column(Enum(PaymentStatus), nullable=False)
    stripe_payment_id = Column(String)
    
    # Transaction details
    payment_date = Column(DateTime, nullable=False)
    description = Column(String)
    metadata = Column(JSON)
    
    # Relationships
    subscription = relationship("Subscription", back_populates="payments")
    user = relationship("User", back_populates="billing_history")

class BillingHistory(Base):
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    
    # Transaction details
    amount = Column(Float, nullable=False)
    currency = Column(String, nullable=False)
    description = Column(String)
    transaction_date = Column(DateTime, nullable=False)
    
    # Token transactions
    tokens_purchased = Column(Float)
    tokens_used = Column(Float)
    token_balance = Column(Float)
    
    # Additional data
    metadata = Column(JSON)
    
    # Relationships
    user = relationship("User", back_populates="billing_history") 