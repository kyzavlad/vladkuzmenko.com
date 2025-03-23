from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey, JSON, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from .base import Base

class TokenPlanDB(Base):
    __tablename__ = "token_plans"

    id = Column(String(36), primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    token_amount = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    currency = Column(String(3), default="USD")
    is_subscription = Column(Boolean, default=False)
    subscription_period = Column(String(10))
    features = Column(JSON)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class TokenTransactionDB(Base):
    __tablename__ = "token_transactions"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), nullable=False)
    amount = Column(Integer, nullable=False)
    transaction_type = Column(String(20), nullable=False)
    plan_id = Column(String(36), ForeignKey("token_plans.id"))
    job_id = Column(String(36), ForeignKey("jobs.id"))
    description = Column(Text)
    metadata = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    plan = relationship("TokenPlanDB", back_populates="transactions")
    job = relationship("JobDB", back_populates="token_transactions")

class TokenBalanceDB(Base):
    __tablename__ = "token_balances"

    user_id = Column(String(36), primary_key=True)
    balance = Column(Integer, nullable=False)
    total_earned = Column(Integer, nullable=False)
    total_spent = Column(Integer, nullable=False)
    last_updated = Column(DateTime(timezone=True), onupdate=func.now())

class TokenUsageDB(Base):
    __tablename__ = "token_usage"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), nullable=False)
    job_id = Column(String(36), ForeignKey("jobs.id"), nullable=False)
    tokens_used = Column(Integer, nullable=False)
    duration = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    job = relationship("JobDB", back_populates="token_usage")

class TokenNotificationDB(Base):
    __tablename__ = "token_notifications"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), nullable=False)
    type = Column(String(20), nullable=False)
    message = Column(Text, nullable=False)
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class TokenPromotionDB(Base):
    __tablename__ = "token_promotions"

    id = Column(String(36), primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    token_amount = Column(Integer, nullable=False)
    start_date = Column(DateTime(timezone=True), nullable=False)
    end_date = Column(DateTime(timezone=True), nullable=False)
    conditions = Column(JSON)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class TokenSubscriptionDB(Base):
    __tablename__ = "token_subscriptions"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), nullable=False)
    plan_id = Column(String(36), ForeignKey("token_plans.id"), nullable=False)
    status = Column(String(20), nullable=False)
    start_date = Column(DateTime(timezone=True), nullable=False)
    end_date = Column(DateTime(timezone=True), nullable=False)
    auto_renew = Column(Boolean, default=True)
    payment_method_id = Column(String(100), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    plan = relationship("TokenPlanDB", back_populates="subscriptions")

class TokenPurchaseDB(Base):
    __tablename__ = "token_purchases"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), nullable=False)
    plan_id = Column(String(36), ForeignKey("token_plans.id"), nullable=False)
    amount = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    currency = Column(String(3), default="USD")
    payment_intent_id = Column(String(100), nullable=False)
    status = Column(String(20), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    plan = relationship("TokenPlanDB", back_populates="purchases")

class TokenAnalyticsDB(Base):
    __tablename__ = "token_analytics"

    user_id = Column(String(36), primary_key=True)
    total_tokens_used = Column(Integer, nullable=False)
    total_duration = Column(Float, nullable=False)
    average_duration = Column(Float, nullable=False)
    most_used_features = Column(JSON)
    last_30_days_usage = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now()) 