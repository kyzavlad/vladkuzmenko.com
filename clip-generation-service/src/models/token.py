from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from enum import Enum

class TokenTransactionType(str, Enum):
    PURCHASE = "purchase"
    USAGE = "usage"
    PROMOTION = "promotion"
    REFUND = "refund"
    EXPIRATION = "expiration"
    RENEWAL = "renewal"

class TokenPlan(BaseModel):
    id: str
    name: str
    description: str
    token_amount: int
    price: float
    currency: str = "USD"
    is_subscription: bool = False
    subscription_period: Optional[str] = None  # monthly, yearly
    features: List[str] = []
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class TokenTransaction(BaseModel):
    id: str
    user_id: str
    amount: int
    transaction_type: TokenTransactionType
    plan_id: Optional[str] = None
    job_id: Optional[str] = None
    description: Optional[str] = None
    metadata: dict = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)

class TokenBalance(BaseModel):
    user_id: str
    balance: int
    total_earned: int
    total_spent: int
    last_updated: datetime = Field(default_factory=datetime.utcnow)

class TokenUsage(BaseModel):
    user_id: str
    job_id: str
    tokens_used: int
    duration: float  # in seconds
    created_at: datetime = Field(default_factory=datetime.utcnow)

class TokenNotification(BaseModel):
    user_id: str
    type: str  # low_balance, expiration, renewal
    message: str
    is_read: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)

class TokenPromotion(BaseModel):
    id: str
    name: str
    description: str
    token_amount: int
    start_date: datetime
    end_date: datetime
    conditions: dict = {}
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class TokenSubscription(BaseModel):
    user_id: str
    plan_id: str
    status: str  # active, cancelled, expired
    start_date: datetime
    end_date: datetime
    auto_renew: bool = True
    payment_method_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class TokenPurchase(BaseModel):
    id: str
    user_id: str
    plan_id: str
    amount: int
    price: float
    currency: str = "USD"
    payment_intent_id: str
    status: str  # pending, completed, failed, refunded
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class TokenAnalytics(BaseModel):
    user_id: str
    total_tokens_used: int
    total_duration: float
    average_duration: float
    most_used_features: List[str]
    last_30_days_usage: dict
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow) 