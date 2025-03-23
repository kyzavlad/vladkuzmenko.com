"""create token tables

Revision ID: create_token_tables
Revises: previous_revision
Create Date: 2024-03-21 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from datetime import datetime

# revision identifiers, used by Alembic.
revision = 'create_token_tables'
down_revision = 'previous_revision'
branch_labels = None
depends_on = None

def upgrade():
    # Create token_plans table
    op.create_table(
        'token_plans',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('token_amount', sa.Integer, nullable=False),
        sa.Column('price', sa.Float, nullable=False),
        sa.Column('currency', sa.String(3), default='USD'),
        sa.Column('is_subscription', sa.Boolean, default=False),
        sa.Column('subscription_period', sa.String(10)),
        sa.Column('features', sa.JSON),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('created_at', sa.DateTime, default=datetime.utcnow),
        sa.Column('updated_at', sa.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    )

    # Create token_transactions table
    op.create_table(
        'token_transactions',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), nullable=False),
        sa.Column('amount', sa.Integer, nullable=False),
        sa.Column('transaction_type', sa.String(20), nullable=False),
        sa.Column('plan_id', sa.String(36), sa.ForeignKey('token_plans.id')),
        sa.Column('job_id', sa.String(36), sa.ForeignKey('jobs.id')),
        sa.Column('description', sa.Text),
        sa.Column('metadata', sa.JSON),
        sa.Column('created_at', sa.DateTime, default=datetime.utcnow)
    )

    # Create token_balances table
    op.create_table(
        'token_balances',
        sa.Column('user_id', sa.String(36), primary_key=True),
        sa.Column('balance', sa.Integer, nullable=False),
        sa.Column('total_earned', sa.Integer, nullable=False),
        sa.Column('total_spent', sa.Integer, nullable=False),
        sa.Column('last_updated', sa.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    )

    # Create token_usage table
    op.create_table(
        'token_usage',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), nullable=False),
        sa.Column('job_id', sa.String(36), sa.ForeignKey('jobs.id'), nullable=False),
        sa.Column('tokens_used', sa.Integer, nullable=False),
        sa.Column('duration', sa.Float, nullable=False),
        sa.Column('created_at', sa.DateTime, default=datetime.utcnow)
    )

    # Create token_notifications table
    op.create_table(
        'token_notifications',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), nullable=False),
        sa.Column('type', sa.String(20), nullable=False),
        sa.Column('message', sa.Text, nullable=False),
        sa.Column('is_read', sa.Boolean, default=False),
        sa.Column('created_at', sa.DateTime, default=datetime.utcnow)
    )

    # Create token_promotions table
    op.create_table(
        'token_promotions',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('token_amount', sa.Integer, nullable=False),
        sa.Column('start_date', sa.DateTime, nullable=False),
        sa.Column('end_date', sa.DateTime, nullable=False),
        sa.Column('conditions', sa.JSON),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('created_at', sa.DateTime, default=datetime.utcnow),
        sa.Column('updated_at', sa.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    )

    # Create token_subscriptions table
    op.create_table(
        'token_subscriptions',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), nullable=False),
        sa.Column('plan_id', sa.String(36), sa.ForeignKey('token_plans.id'), nullable=False),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('start_date', sa.DateTime, nullable=False),
        sa.Column('end_date', sa.DateTime, nullable=False),
        sa.Column('auto_renew', sa.Boolean, default=True),
        sa.Column('payment_method_id', sa.String(100), nullable=False),
        sa.Column('created_at', sa.DateTime, default=datetime.utcnow),
        sa.Column('updated_at', sa.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    )

    # Create token_purchases table
    op.create_table(
        'token_purchases',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), nullable=False),
        sa.Column('plan_id', sa.String(36), sa.ForeignKey('token_plans.id'), nullable=False),
        sa.Column('amount', sa.Integer, nullable=False),
        sa.Column('price', sa.Float, nullable=False),
        sa.Column('currency', sa.String(3), default='USD'),
        sa.Column('payment_intent_id', sa.String(100), nullable=False),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('created_at', sa.DateTime, default=datetime.utcnow),
        sa.Column('updated_at', sa.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    )

    # Create token_analytics table
    op.create_table(
        'token_analytics',
        sa.Column('user_id', sa.String(36), primary_key=True),
        sa.Column('total_tokens_used', sa.Integer, nullable=False),
        sa.Column('total_duration', sa.Float, nullable=False),
        sa.Column('average_duration', sa.Float, nullable=False),
        sa.Column('most_used_features', sa.JSON),
        sa.Column('last_30_days_usage', sa.JSON),
        sa.Column('created_at', sa.DateTime, default=datetime.utcnow),
        sa.Column('updated_at', sa.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    )

    # Create indexes
    op.create_index('idx_token_transactions_user_id', 'token_transactions', ['user_id'])
    op.create_index('idx_token_transactions_job_id', 'token_transactions', ['job_id'])
    op.create_index('idx_token_usage_user_id', 'token_usage', ['user_id'])
    op.create_index('idx_token_usage_job_id', 'token_usage', ['job_id'])
    op.create_index('idx_token_notifications_user_id', 'token_notifications', ['user_id'])
    op.create_index('idx_token_subscriptions_user_id', 'token_subscriptions', ['user_id'])
    op.create_index('idx_token_purchases_user_id', 'token_purchases', ['user_id'])

def downgrade():
    # Drop indexes
    op.drop_index('idx_token_transactions_user_id')
    op.drop_index('idx_token_transactions_job_id')
    op.drop_index('idx_token_usage_user_id')
    op.drop_index('idx_token_usage_job_id')
    op.drop_index('idx_token_notifications_user_id')
    op.drop_index('idx_token_subscriptions_user_id')
    op.drop_index('idx_token_purchases_user_id')

    # Drop tables
    op.drop_table('token_analytics')
    op.drop_table('token_purchases')
    op.drop_table('token_subscriptions')
    op.drop_table('token_promotions')
    op.drop_table('token_notifications')
    op.drop_table('token_usage')
    op.drop_table('token_balances')
    op.drop_table('token_transactions')
    op.drop_table('token_plans') 