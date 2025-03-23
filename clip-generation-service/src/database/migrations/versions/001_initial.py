"""initial

Revision ID: 001
Revises: 
Create Date: 2024-03-21 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(), nullable=False),
        sa.Column('email', sa.String(), nullable=False),
        sa.Column('hashed_password', sa.String(), nullable=False),
        sa.Column('full_name', sa.String(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('is_superuser', sa.Boolean(), nullable=False, default=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('username'),
        sa.UniqueConstraint('email')
    )
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)

    # Create jobs table
    op.create_table(
        'jobs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('job_id', sa.String(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('job_type', sa.String(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('progress', sa.Float(), nullable=False, default=0.0),
        sa.Column('input_data', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('output_data', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('error_message', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('job_id')
    )
    op.create_index(op.f('ix_jobs_id'), 'jobs', ['id'], unique=False)
    op.create_index(op.f('ix_jobs_job_id'), 'jobs', ['job_id'], unique=True)
    op.create_index(op.f('ix_jobs_user_id'), 'jobs', ['user_id'], unique=False)

    # Create job_logs table
    op.create_table(
        'job_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('job_id', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('level', sa.String(), nullable=False),
        sa.Column('message', sa.String(), nullable=False),
        sa.Column('details', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['job_id'], ['jobs.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_job_logs_id'), 'job_logs', ['id'], unique=False)
    op.create_index(op.f('ix_job_logs_job_id'), 'job_logs', ['job_id'], unique=False)

    # Create api_keys table
    op.create_table(
        'api_keys',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('key', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('description', sa.String(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('last_used_at', sa.DateTime(), nullable=True),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('permissions', postgresql.JSON(astext_type=sa.Text()), nullable=False, default=list),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('key')
    )
    op.create_index(op.f('ix_api_keys_id'), 'api_keys', ['id'], unique=False)
    op.create_index(op.f('ix_api_keys_key'), 'api_keys', ['key'], unique=True)
    op.create_index(op.f('ix_api_keys_user_id'), 'api_keys', ['user_id'], unique=False)

    # Create security_audits table
    op.create_table(
        'security_audits',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('action', sa.String(), nullable=False),
        sa.Column('ip_address', sa.String(), nullable=False),
        sa.Column('user_agent', sa.String(), nullable=False),
        sa.Column('details', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('security_level', sa.String(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_security_audits_id'), 'security_audits', ['id'], unique=False)
    op.create_index(op.f('ix_security_audits_user_id'), 'security_audits', ['user_id'], unique=False)

def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_index(op.f('ix_security_audits_user_id'), table_name='security_audits')
    op.drop_index(op.f('ix_security_audits_id'), table_name='security_audits')
    op.drop_table('security_audits')

    op.drop_index(op.f('ix_api_keys_user_id'), table_name='api_keys')
    op.drop_index(op.f('ix_api_keys_key'), table_name='api_keys')
    op.drop_index(op.f('ix_api_keys_id'), table_name='api_keys')
    op.drop_table('api_keys')

    op.drop_index(op.f('ix_job_logs_job_id'), table_name='job_logs')
    op.drop_index(op.f('ix_job_logs_id'), table_name='job_logs')
    op.drop_table('job_logs')

    op.drop_index(op.f('ix_jobs_user_id'), table_name='jobs')
    op.drop_index(op.f('ix_jobs_job_id'), table_name='jobs')
    op.drop_index(op.f('ix_jobs_id'), table_name='jobs')
    op.drop_table('jobs')

    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_index(op.f('ix_users_username'), table_name='users')
    op.drop_index(op.f('ix_users_id'), table_name='users')
    op.drop_table('users') 