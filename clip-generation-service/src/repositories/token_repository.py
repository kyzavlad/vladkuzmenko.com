from typing import List, Optional, Dict
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from uuid import uuid4

from ..models.token import (
    TokenPlan, TokenTransaction, TokenBalance, TokenUsage,
    TokenNotification, TokenPromotion, TokenSubscription,
    TokenPurchase, TokenAnalytics
)
from ..database.models import (
    TokenPlanDB, TokenTransactionDB, TokenBalanceDB, TokenUsageDB,
    TokenNotificationDB, TokenPromotionDB, TokenSubscriptionDB,
    TokenPurchaseDB, TokenAnalyticsDB
)

class TokenRepository:
    def __init__(self, db: Session):
        self.db = db

    # Token Plan Operations
    def create_plan(self, plan: TokenPlan) -> TokenPlan:
        db_plan = TokenPlanDB(**plan.dict())
        self.db.add(db_plan)
        self.db.commit()
        self.db.refresh(db_plan)
        return TokenPlan.from_orm(db_plan)

    def get_plan(self, plan_id: str) -> Optional[TokenPlan]:
        db_plan = self.db.query(TokenPlanDB).filter(TokenPlanDB.id == plan_id).first()
        return TokenPlan.from_orm(db_plan) if db_plan else None

    def get_active_plans(self) -> List[TokenPlan]:
        db_plans = self.db.query(TokenPlanDB).filter(TokenPlanDB.is_active == True).all()
        return [TokenPlan.from_orm(plan) for plan in db_plans]

    # Token Balance Operations
    def get_balance(self, user_id: str) -> Optional[TokenBalance]:
        db_balance = self.db.query(TokenBalanceDB).filter(TokenBalanceDB.user_id == user_id).first()
        return TokenBalance.from_orm(db_balance) if db_balance else None

    def create_balance(self, user_id: str, initial_balance: int = 60) -> TokenBalance:
        db_balance = TokenBalanceDB(
            user_id=user_id,
            balance=initial_balance,
            total_earned=initial_balance,
            total_spent=0
        )
        self.db.add(db_balance)
        self.db.commit()
        self.db.refresh(db_balance)
        return TokenBalance.from_orm(db_balance)

    def update_balance(self, user_id: str, amount: int, transaction_type: str) -> TokenBalance:
        db_balance = self.db.query(TokenBalanceDB).filter(TokenBalanceDB.user_id == user_id).first()
        if not db_balance:
            db_balance = self.create_balance(user_id)

        if transaction_type == "earn":
            db_balance.balance += amount
            db_balance.total_earned += amount
        elif transaction_type == "spend":
            db_balance.balance -= amount
            db_balance.total_spent += amount

        self.db.commit()
        self.db.refresh(db_balance)
        return TokenBalance.from_orm(db_balance)

    # Token Transaction Operations
    def create_transaction(self, transaction: TokenTransaction) -> TokenTransaction:
        db_transaction = TokenTransactionDB(**transaction.dict())
        self.db.add(db_transaction)
        self.db.commit()
        self.db.refresh(db_transaction)
        return TokenTransaction.from_orm(db_transaction)

    def get_user_transactions(
        self,
        user_id: str,
        limit: int = 10,
        offset: int = 0
    ) -> List[TokenTransaction]:
        db_transactions = self.db.query(TokenTransactionDB)\
            .filter(TokenTransactionDB.user_id == user_id)\
            .order_by(TokenTransactionDB.created_at.desc())\
            .offset(offset)\
            .limit(limit)\
            .all()
        return [TokenTransaction.from_orm(t) for t in db_transactions]

    # Token Usage Operations
    def create_usage(self, usage: TokenUsage) -> TokenUsage:
        db_usage = TokenUsageDB(**usage.dict())
        self.db.add(db_usage)
        self.db.commit()
        self.db.refresh(db_usage)
        return TokenUsage.from_orm(db_usage)

    def get_user_usage(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[TokenUsage]:
        query = self.db.query(TokenUsageDB).filter(TokenUsageDB.user_id == user_id)
        
        if start_date:
            query = query.filter(TokenUsageDB.created_at >= start_date)
        if end_date:
            query = query.filter(TokenUsageDB.created_at <= end_date)
            
        db_usages = query.order_by(TokenUsageDB.created_at.desc()).all()
        return [TokenUsage.from_orm(u) for u in db_usages]

    # Token Notification Operations
    def create_notification(self, notification: TokenNotification) -> TokenNotification:
        db_notification = TokenNotificationDB(**notification.dict())
        self.db.add(db_notification)
        self.db.commit()
        self.db.refresh(db_notification)
        return TokenNotification.from_orm(db_notification)

    def get_unread_notifications(self, user_id: str) -> List[TokenNotification]:
        db_notifications = self.db.query(TokenNotificationDB)\
            .filter(
                and_(
                    TokenNotificationDB.user_id == user_id,
                    TokenNotificationDB.is_read == False
                )
            )\
            .order_by(TokenNotificationDB.created_at.desc())\
            .all()
        return [TokenNotification.from_orm(n) for n in db_notifications]

    def mark_notification_read(self, notification_id: str) -> TokenNotification:
        db_notification = self.db.query(TokenNotificationDB)\
            .filter(TokenNotificationDB.id == notification_id)\
            .first()
        if db_notification:
            db_notification.is_read = True
            self.db.commit()
            self.db.refresh(db_notification)
        return TokenNotification.from_orm(db_notification) if db_notification else None

    # Token Promotion Operations
    def create_promotion(self, promotion: TokenPromotion) -> TokenPromotion:
        db_promotion = TokenPromotionDB(**promotion.dict())
        self.db.add(db_promotion)
        self.db.commit()
        self.db.refresh(db_promotion)
        return TokenPromotion.from_orm(db_promotion)

    def get_active_promotions(self) -> List[TokenPromotion]:
        now = datetime.utcnow()
        db_promotions = self.db.query(TokenPromotionDB)\
            .filter(
                and_(
                    TokenPromotionDB.is_active == True,
                    TokenPromotionDB.start_date <= now,
                    TokenPromotionDB.end_date >= now
                )
            )\
            .all()
        return [TokenPromotion.from_orm(p) for p in db_promotions]

    # Token Subscription Operations
    def create_subscription(self, subscription: TokenSubscription) -> TokenSubscription:
        db_subscription = TokenSubscriptionDB(**subscription.dict())
        self.db.add(db_subscription)
        self.db.commit()
        self.db.refresh(db_subscription)
        return TokenSubscription.from_orm(db_subscription)

    def get_active_subscription(self, user_id: str) -> Optional[TokenSubscription]:
        now = datetime.utcnow()
        db_subscription = self.db.query(TokenSubscriptionDB)\
            .filter(
                and_(
                    TokenSubscriptionDB.user_id == user_id,
                    TokenSubscriptionDB.status == "active",
                    TokenSubscriptionDB.end_date >= now
                )
            )\
            .first()
        return TokenSubscription.from_orm(db_subscription) if db_subscription else None

    # Token Purchase Operations
    def create_purchase(self, purchase: TokenPurchase) -> TokenPurchase:
        db_purchase = TokenPurchaseDB(**purchase.dict())
        self.db.add(db_purchase)
        self.db.commit()
        self.db.refresh(db_purchase)
        return TokenPurchase.from_orm(db_purchase)

    def get_purchase(self, purchase_id: str) -> Optional[TokenPurchase]:
        db_purchase = self.db.query(TokenPurchaseDB).filter(TokenPurchaseDB.id == purchase_id).first()
        return TokenPurchase.from_orm(db_purchase) if db_purchase else None

    # Token Analytics Operations
    def get_analytics(self, user_id: str) -> Optional[TokenAnalytics]:
        db_analytics = self.db.query(TokenAnalyticsDB).filter(TokenAnalyticsDB.user_id == user_id).first()
        return TokenAnalytics.from_orm(db_analytics) if db_analytics else None

    def update_analytics(self, user_id: str, usage_data: Dict) -> TokenAnalytics:
        db_analytics = self.db.query(TokenAnalyticsDB).filter(TokenAnalyticsDB.user_id == user_id).first()
        if not db_analytics:
            db_analytics = TokenAnalyticsDB(user_id=user_id, **usage_data)
            self.db.add(db_analytics)
        else:
            for key, value in usage_data.items():
                setattr(db_analytics, key, value)
        
        self.db.commit()
        self.db.refresh(db_analytics)
        return TokenAnalytics.from_orm(db_analytics)

    def get_usage_statistics(self, user_id: str) -> Dict:
        # Get last 30 days usage
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        usage_data = self.db.query(
            func.date(TokenUsageDB.created_at).label('date'),
            func.sum(TokenUsageDB.tokens_used).label('total_tokens'),
            func.sum(TokenUsageDB.duration).label('total_duration')
        ).filter(
            and_(
                TokenUsageDB.user_id == user_id,
                TokenUsageDB.created_at >= thirty_days_ago
            )
        ).group_by(
            func.date(TokenUsageDB.created_at)
        ).all()

        return {
            'daily_usage': [
                {
                    'date': str(usage.date),
                    'tokens': usage.total_tokens,
                    'duration': usage.total_duration
                }
                for usage in usage_data
            ]
        } 