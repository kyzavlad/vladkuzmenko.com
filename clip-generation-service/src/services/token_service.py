from typing import List, Optional, Dict
from datetime import datetime, timedelta
from uuid import uuid4
import stripe
from fastapi import HTTPException, status

from ..repositories.token_repository import TokenRepository
from ..models.token import (
    TokenPlan, TokenTransaction, TokenBalance, TokenUsage,
    TokenNotification, TokenPromotion, TokenSubscription,
    TokenPurchase, TokenAnalytics
)

class TokenService:
    def __init__(self, token_repository: TokenRepository, stripe_secret_key: str):
        self.token_repository = token_repository
        stripe.api_key = stripe_secret_key
        self.FREE_TOKENS = 60
        self.TOKENS_PER_SECOND = 1

    # Token Balance Management
    async def get_balance(self, user_id: str) -> TokenBalance:
        balance = self.token_repository.get_balance(user_id)
        if not balance:
            balance = self.token_repository.create_balance(user_id, self.FREE_TOKENS)
        return balance

    async def check_balance(self, user_id: str, required_tokens: int) -> bool:
        balance = await self.get_balance(user_id)
        return balance.balance >= required_tokens

    async def deduct_tokens(self, user_id: str, amount: int, job_id: str) -> TokenBalance:
        if not await self.check_balance(user_id, amount):
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail="Insufficient token balance"
            )

        # Create transaction
        transaction = TokenTransaction(
            id=str(uuid4()),
            user_id=user_id,
            amount=amount,
            transaction_type="USAGE",
            job_id=job_id,
            description=f"Token usage for job {job_id}"
        )
        self.token_repository.create_transaction(transaction)

        # Update balance
        return self.token_repository.update_balance(user_id, amount, "spend")

    async def add_tokens(self, user_id: str, amount: int, transaction_type: str) -> TokenBalance:
        # Create transaction
        transaction = TokenTransaction(
            id=str(uuid4()),
            user_id=user_id,
            amount=amount,
            transaction_type=transaction_type,
            description=f"Token addition: {transaction_type}"
        )
        self.token_repository.create_transaction(transaction)

        # Update balance
        return self.token_repository.update_balance(user_id, amount, "earn")

    # Token Usage Tracking
    async def track_usage(self, user_id: str, job_id: str, duration: float) -> TokenUsage:
        tokens_used = int(duration * self.TOKENS_PER_SECOND)
        usage = TokenUsage(
            id=str(uuid4()),
            user_id=user_id,
            job_id=job_id,
            tokens_used=tokens_used,
            duration=duration
        )
        return self.token_repository.create_usage(usage)

    # Token Purchase
    async def create_purchase_intent(
        self,
        user_id: str,
        plan_id: str,
        payment_method_id: str
    ) -> Dict:
        plan = self.token_repository.get_plan(plan_id)
        if not plan:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Plan not found"
            )

        # Create Stripe payment intent
        payment_intent = stripe.PaymentIntent.create(
            amount=int(plan.price * 100),  # Convert to cents
            currency=plan.currency.lower(),
            payment_method=payment_method_id,
            confirm=True,
            customer=user_id  # Assuming user_id is the Stripe customer ID
        )

        # Create purchase record
        purchase = TokenPurchase(
            id=str(uuid4()),
            user_id=user_id,
            plan_id=plan_id,
            amount=plan.token_amount,
            price=plan.price,
            currency=plan.currency,
            payment_intent_id=payment_intent.id,
            status="pending"
        )
        self.token_repository.create_purchase(purchase)

        return {
            "purchase_id": purchase.id,
            "client_secret": payment_intent.client_secret
        }

    async def confirm_purchase(self, purchase_id: str) -> TokenBalance:
        purchase = self.token_repository.get_purchase(purchase_id)
        if not purchase:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Purchase not found"
            )

        # Update purchase status
        purchase.status = "completed"
        self.token_repository.create_purchase(purchase)

        # Add tokens to user balance
        return await self.add_tokens(purchase.user_id, purchase.amount, "PURCHASE")

    # Subscription Management
    async def create_subscription(
        self,
        user_id: str,
        plan_id: str,
        payment_method_id: str
    ) -> TokenSubscription:
        plan = self.token_repository.get_plan(plan_id)
        if not plan or not plan.is_subscription:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid subscription plan"
            )

        # Create Stripe subscription
        subscription = stripe.Subscription.create(
            customer=user_id,
            items=[{"price": plan.stripe_price_id}],
            payment_behavior="default_incomplete",
            payment_settings={"save_default_payment_method": "on_subscription"},
            expand=["latest_invoice.payment_intent"]
        )

        # Create subscription record
        db_subscription = TokenSubscription(
            id=str(uuid4()),
            user_id=user_id,
            plan_id=plan_id,
            status="active",
            start_date=datetime.fromtimestamp(subscription.start_date),
            end_date=datetime.fromtimestamp(subscription.current_period_end),
            auto_renew=True,
            payment_method_id=payment_method_id
        )
        return self.token_repository.create_subscription(db_subscription)

    async def cancel_subscription(self, user_id: str) -> TokenSubscription:
        subscription = self.token_repository.get_active_subscription(user_id)
        if not subscription:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No active subscription found"
            )

        # Cancel Stripe subscription
        stripe.Subscription.delete(subscription.stripe_subscription_id)

        # Update subscription status
        subscription.status = "cancelled"
        subscription.auto_renew = False
        return self.token_repository.create_subscription(subscription)

    # Promotion Management
    async def apply_promotion(self, user_id: str, promotion_code: str) -> TokenBalance:
        promotion = self.token_repository.get_active_promotions()
        if not promotion:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invalid or expired promotion code"
            )

        # Check if user has already used this promotion
        if await self.has_used_promotion(user_id, promotion.id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Promotion already used"
            )

        # Add promotion tokens
        return await self.add_tokens(user_id, promotion.token_amount, "PROMOTION")

    async def has_used_promotion(self, user_id: str, promotion_id: str) -> bool:
        transactions = self.token_repository.get_user_transactions(user_id)
        return any(
            t.transaction_type == "PROMOTION" and t.metadata.get("promotion_id") == promotion_id
            for t in transactions
        )

    # Notification Management
    async def check_low_balance(self, user_id: str) -> Optional[TokenNotification]:
        balance = await self.get_balance(user_id)
        if balance.balance < 10:  # Threshold for low balance
            notification = TokenNotification(
                id=str(uuid4()),
                user_id=user_id,
                type="low_balance",
                message=f"Your token balance is low ({balance.balance} tokens remaining)"
            )
            return self.token_repository.create_notification(notification)
        return None

    async def get_notifications(self, user_id: str) -> List[TokenNotification]:
        return self.token_repository.get_unread_notifications(user_id)

    async def mark_notification_read(self, notification_id: str) -> TokenNotification:
        return self.token_repository.mark_notification_read(notification_id)

    # Analytics
    async def update_analytics(self, user_id: str) -> TokenAnalytics:
        usage_data = self.token_repository.get_usage_statistics(user_id)
        total_usage = sum(day["tokens"] for day in usage_data["daily_usage"])
        total_duration = sum(day["duration"] for day in usage_data["daily_usage"])
        average_duration = total_duration / len(usage_data["daily_usage"]) if usage_data["daily_usage"] else 0

        analytics_data = {
            "total_tokens_used": total_usage,
            "total_duration": total_duration,
            "average_duration": average_duration,
            "most_used_features": [],  # To be implemented based on feature tracking
            "last_30_days_usage": usage_data
        }

        return self.token_repository.update_analytics(user_id, analytics_data)

    async def get_analytics(self, user_id: str) -> Optional[TokenAnalytics]:
        return self.token_repository.get_analytics(user_id) 