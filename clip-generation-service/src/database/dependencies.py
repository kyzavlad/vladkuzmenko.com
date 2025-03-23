from typing import Generator
from fastapi import Depends
from sqlalchemy.orm import Session
from .connection import get_db
from .repositories import (
    UserRepository,
    JobRepository,
    JobLogRepository,
    ApiKeyRepository,
    SecurityAuditRepository
)

def get_db_session() -> Generator[Session, None, None]:
    """Get database session."""
    db = get_db()
    with db.get_session() as session:
        yield session

def get_user_repository(
    session: Session = Depends(get_db_session)
) -> UserRepository:
    """Get user repository."""
    return UserRepository(session)

def get_job_repository(
    session: Session = Depends(get_db_session)
) -> JobRepository:
    """Get job repository."""
    return JobRepository(session)

def get_job_log_repository(
    session: Session = Depends(get_db_session)
) -> JobLogRepository:
    """Get job log repository."""
    return JobLogRepository(session)

def get_api_key_repository(
    session: Session = Depends(get_db_session)
) -> ApiKeyRepository:
    """Get API key repository."""
    return ApiKeyRepository(session)

def get_security_audit_repository(
    session: Session = Depends(get_db_session)
) -> SecurityAuditRepository:
    """Get security audit repository."""
    return SecurityAuditRepository(session) 