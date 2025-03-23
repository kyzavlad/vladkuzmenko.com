from fastapi import APIRouter, Request, HTTPException, status
from sqlalchemy.orm import Session
import stripe
from typing import Optional

from ..database.session import get_db
from ..services.token_service import TokenService
from ..repositories.token_repository import TokenRepository
from ..config import settings
from ..models.token import TokenPurchase, TokenSubscription

router = APIRouter(prefix="/webhook", tags=["webhook"])

def get_token_service(db: Session = Depends(get_db)) -> TokenService:
    token_repository = TokenRepository(db)
    return TokenService(token_repository, settings.STRIPE_SECRET_KEY)

@router.post("/stripe")
async def stripe_webhook(
    request: Request,
    token_service: TokenService = Depends(get_token_service)
):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, settings.STRIPE_WEBHOOK_SECRET
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid payload"
        )
    except stripe.error.SignatureVerificationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid signature"
        )

    # Handle the event
    if event.type == "payment_intent.succeeded":
        payment_intent = event.data.object
        await handle_payment_succeeded(payment_intent, token_service)
    elif event.type == "payment_intent.payment_failed":
        payment_intent = event.data.object
        await handle_payment_failed(payment_intent, token_service)
    elif event.type == "customer.subscription.created":
        subscription = event.data.object
        await handle_subscription_created(subscription, token_service)
    elif event.type == "customer.subscription.updated":
        subscription = event.data.object
        await handle_subscription_updated(subscription, token_service)
    elif event.type == "customer.subscription.deleted":
        subscription = event.data.object
        await handle_subscription_deleted(subscription, token_service)
    elif event.type == "invoice.payment_succeeded":
        invoice = event.data.object
        await handle_invoice_payment_succeeded(invoice, token_service)
    elif event.type == "invoice.payment_failed":
        invoice = event.data.object
        await handle_invoice_payment_failed(invoice, token_service)

    return {"status": "success"}

async def handle_payment_succeeded(payment_intent, token_service: TokenService):
    # Find the purchase record
    purchase = token_service.token_repository.get_purchase_by_payment_intent(payment_intent.id)
    if purchase:
        # Update purchase status and add tokens
        await token_service.confirm_purchase(purchase.id)

async def handle_payment_failed(payment_intent, token_service: TokenService):
    # Find the purchase record
    purchase = token_service.token_repository.get_purchase_by_payment_intent(payment_intent.id)
    if purchase:
        # Update purchase status to failed
        purchase.status = "failed"
        token_service.token_repository.create_purchase(purchase)

async def handle_subscription_created(subscription, token_service: TokenService):
    # Create subscription record
    db_subscription = TokenSubscription(
        id=str(uuid4()),
        user_id=subscription.customer,
        plan_id=subscription.items.data[0].price.id,
        status="active",
        start_date=datetime.fromtimestamp(subscription.start_date),
        end_date=datetime.fromtimestamp(subscription.current_period_end),
        auto_renew=True,
        payment_method_id=subscription.default_payment_method
    )
    token_service.token_repository.create_subscription(db_subscription)

async def handle_subscription_updated(subscription, token_service: TokenService):
    # Update subscription record
    db_subscription = token_service.token_repository.get_subscription_by_stripe_id(subscription.id)
    if db_subscription:
        db_subscription.status = subscription.status
        db_subscription.end_date = datetime.fromtimestamp(subscription.current_period_end)
        db_subscription.auto_renew = subscription.cancel_at_period_end is None
        token_service.token_repository.create_subscription(db_subscription)

async def handle_subscription_deleted(subscription, token_service: TokenService):
    # Update subscription record
    db_subscription = token_service.token_repository.get_subscription_by_stripe_id(subscription.id)
    if db_subscription:
        db_subscription.status = "cancelled"
        db_subscription.auto_renew = False
        token_service.token_repository.create_subscription(db_subscription)

async def handle_invoice_payment_succeeded(invoice, token_service: TokenService):
    # Handle subscription renewal
    if invoice.subscription:
        subscription = token_service.token_repository.get_subscription_by_stripe_id(invoice.subscription)
        if subscription:
            # Add tokens for the new period
            plan = token_service.token_repository.get_plan(subscription.plan_id)
            if plan:
                await token_service.add_tokens(
                    subscription.user_id,
                    plan.token_amount,
                    "SUBSCRIPTION_RENEWAL"
                )

async def handle_invoice_payment_failed(invoice, token_service: TokenService):
    # Handle subscription payment failure
    if invoice.subscription:
        subscription = token_service.token_repository.get_subscription_by_stripe_id(invoice.subscription)
        if subscription:
            # Update subscription status
            subscription.status = "past_due"
            token_service.token_repository.create_subscription(subscription)

            # Send notification to user
            notification = TokenNotification(
                id=str(uuid4()),
                user_id=subscription.user_id,
                type="payment_failed",
                message="Your subscription payment has failed. Please update your payment method."
            )
            token_service.token_repository.create_notification(notification) 