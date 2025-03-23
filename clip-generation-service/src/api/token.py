from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..database.session import get_db
from ..services.token_service import TokenService
from ..repositories.token_repository import TokenRepository
from ..models.token import (
    TokenPlan, TokenTransaction, TokenBalance, TokenUsage,
    TokenNotification, TokenPromotion, TokenSubscription,
    TokenPurchase, TokenAnalytics
)
from ..auth import get_current_user
from ..config import settings

router = APIRouter(prefix="/tokens", tags=["tokens"])

def get_token_service(db: Session = Depends(get_db)) -> TokenService:
    token_repository = TokenRepository(db)
    return TokenService(token_repository, settings.STRIPE_SECRET_KEY)

# Token Balance Endpoints
@router.get("/balance", response_model=TokenBalance)
async def get_balance(
    current_user = Depends(get_current_user),
    token_service: TokenService = Depends(get_token_service)
):
    return await token_service.get_balance(current_user.id)

@router.get("/transactions", response_model=List[TokenTransaction])
async def get_transactions(
    limit: int = 10,
    offset: int = 0,
    current_user = Depends(get_current_user),
    token_service: TokenService = Depends(get_token_service)
):
    return token_service.token_repository.get_user_transactions(current_user.id, limit, offset)

# Token Purchase Endpoints
@router.post("/purchase/intent", response_model=dict)
async def create_purchase_intent(
    plan_id: str,
    payment_method_id: str,
    current_user = Depends(get_current_user),
    token_service: TokenService = Depends(get_token_service)
):
    return await token_service.create_purchase_intent(
        current_user.id,
        plan_id,
        payment_method_id
    )

@router.post("/purchase/{purchase_id}/confirm", response_model=TokenBalance)
async def confirm_purchase(
    purchase_id: str,
    current_user = Depends(get_current_user),
    token_service: TokenService = Depends(get_token_service)
):
    return await token_service.confirm_purchase(purchase_id)

# Subscription Endpoints
@router.post("/subscription", response_model=TokenSubscription)
async def create_subscription(
    plan_id: str,
    payment_method_id: str,
    current_user = Depends(get_current_user),
    token_service: TokenService = Depends(get_token_service)
):
    return await token_service.create_subscription(
        current_user.id,
        plan_id,
        payment_method_id
    )

@router.delete("/subscription", response_model=TokenSubscription)
async def cancel_subscription(
    current_user = Depends(get_current_user),
    token_service: TokenService = Depends(get_token_service)
):
    return await token_service.cancel_subscription(current_user.id)

# Promotion Endpoints
@router.post("/promotion/{code}", response_model=TokenBalance)
async def apply_promotion(
    code: str,
    current_user = Depends(get_current_user),
    token_service: TokenService = Depends(get_token_service)
):
    return await token_service.apply_promotion(current_user.id, code)

# Notification Endpoints
@router.get("/notifications", response_model=List[TokenNotification])
async def get_notifications(
    current_user = Depends(get_current_user),
    token_service: TokenService = Depends(get_token_service)
):
    return await token_service.get_notifications(current_user.id)

@router.post("/notifications/{notification_id}/read", response_model=TokenNotification)
async def mark_notification_read(
    notification_id: str,
    current_user = Depends(get_current_user),
    token_service: TokenService = Depends(get_token_service)
):
    return await token_service.mark_notification_read(notification_id)

# Analytics Endpoints
@router.get("/analytics", response_model=TokenAnalytics)
async def get_analytics(
    current_user = Depends(get_current_user),
    token_service: TokenService = Depends(get_token_service)
):
    analytics = await token_service.get_analytics(current_user.id)
    if not analytics:
        analytics = await token_service.update_analytics(current_user.id)
    return analytics

# Admin Endpoints
@router.post("/admin/plans", response_model=TokenPlan)
async def create_plan(
    plan: TokenPlan,
    current_user = Depends(get_current_user),
    token_service: TokenService = Depends(get_token_service)
):
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return token_service.token_repository.create_plan(plan)

@router.post("/admin/promotions", response_model=TokenPromotion)
async def create_promotion(
    promotion: TokenPromotion,
    current_user = Depends(get_current_user),
    token_service: TokenService = Depends(get_token_service)
):
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return token_service.token_repository.create_promotion(promotion)

@router.get("/admin/plans", response_model=List[TokenPlan])
async def get_all_plans(
    current_user = Depends(get_current_user),
    token_service: TokenService = Depends(get_token_service)
):
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return token_service.token_repository.get_active_plans()

@router.get("/admin/promotions", response_model=List[TokenPromotion])
async def get_all_promotions(
    current_user = Depends(get_current_user),
    token_service: TokenService = Depends(get_token_service)
):
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return token_service.token_repository.get_active_promotions() 