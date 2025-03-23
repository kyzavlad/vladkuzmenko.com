from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import select, update, delete
from datetime import datetime
import logging
from ..models.database import User, Job, JobLog, ApiKey, SecurityAudit
from ..models.user import UserCreate, UserUpdate
from ..models.auth import TokenData

logger = logging.getLogger(__name__)

class UserRepository:
    def __init__(self, session: Session):
        self.session = session

    async def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        try:
            result = await self.session.execute(
                select(User).where(User.username == username)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting user by username: {str(e)}")
            raise

    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        try:
            result = await self.session.execute(
                select(User).where(User.email == email)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting user by email: {str(e)}")
            raise

    async def create(self, user_data: Dict[str, Any]) -> User:
        """Create new user."""
        try:
            user = User(**user_data)
            self.session.add(user)
            await self.session.commit()
            await self.session.refresh(user)
            return user
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error creating user: {str(e)}")
            raise

    async def update(self, user_id: int, user_data: Dict[str, Any]) -> Optional[User]:
        """Update user."""
        try:
            result = await self.session.execute(
                update(User)
                .where(User.id == user_id)
                .values(**user_data)
                .returning(User)
            )
            await self.session.commit()
            return result.scalar_one_or_none()
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error updating user: {str(e)}")
            raise

    async def delete(self, user_id: int) -> bool:
        """Delete user."""
        try:
            result = await self.session.execute(
                delete(User)
                .where(User.id == user_id)
                .returning(User.id)
            )
            await self.session.commit()
            return result.scalar_one_or_none() is not None
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error deleting user: {str(e)}")
            raise

class JobRepository:
    def __init__(self, session: Session):
        self.session = session

    async def create(self, job_data: Dict[str, Any]) -> Job:
        """Create new job."""
        try:
            job = Job(**job_data)
            self.session.add(job)
            await self.session.commit()
            await self.session.refresh(job)
            return job
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error creating job: {str(e)}")
            raise

    async def get_by_id(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        try:
            result = await self.session.execute(
                select(Job).where(Job.job_id == job_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting job by ID: {str(e)}")
            raise

    async def get_user_jobs(
        self,
        user_id: int,
        skip: int = 0,
        limit: int = 100,
        job_type: Optional[str] = None
    ) -> List[Job]:
        """Get user's jobs with optional filtering."""
        try:
            query = select(Job).where(Job.user_id == user_id)
            if job_type:
                query = query.where(Job.job_type == job_type)
            query = query.offset(skip).limit(limit)
            result = await self.session.execute(query)
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error getting user jobs: {str(e)}")
            raise

    async def update_status(
        self,
        job_id: str,
        status: str,
        progress: Optional[float] = None,
        error_message: Optional[str] = None
    ) -> Optional[Job]:
        """Update job status."""
        try:
            update_data = {"status": status}
            if progress is not None:
                update_data["progress"] = progress
            if error_message is not None:
                update_data["error_message"] = error_message
            if status == "completed":
                update_data["completed_at"] = datetime.utcnow()

            result = await self.session.execute(
                update(Job)
                .where(Job.job_id == job_id)
                .values(**update_data)
                .returning(Job)
            )
            await self.session.commit()
            return result.scalar_one_or_none()
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error updating job status: {str(e)}")
            raise

class JobLogRepository:
    def __init__(self, session: Session):
        self.session = session

    async def create(self, log_data: Dict[str, Any]) -> JobLog:
        """Create new job log entry."""
        try:
            log = JobLog(**log_data)
            self.session.add(log)
            await self.session.commit()
            await self.session.refresh(log)
            return log
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error creating job log: {str(e)}")
            raise

    async def get_job_logs(
        self,
        job_id: int,
        skip: int = 0,
        limit: int = 100,
        level: Optional[str] = None
    ) -> List[JobLog]:
        """Get job logs with optional filtering."""
        try:
            query = select(JobLog).where(JobLog.job_id == job_id)
            if level:
                query = query.where(JobLog.level == level)
            query = query.order_by(JobLog.timestamp.desc()).offset(skip).limit(limit)
            result = await self.session.execute(query)
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error getting job logs: {str(e)}")
            raise

class ApiKeyRepository:
    def __init__(self, session: Session):
        self.session = session

    async def create(self, api_key_data: Dict[str, Any]) -> ApiKey:
        """Create new API key."""
        try:
            api_key = ApiKey(**api_key_data)
            self.session.add(api_key)
            await self.session.commit()
            await self.session.refresh(api_key)
            return api_key
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error creating API key: {str(e)}")
            raise

    async def get_by_key(self, key: str) -> Optional[ApiKey]:
        """Get API key by key string."""
        try:
            result = await self.session.execute(
                select(ApiKey).where(ApiKey.key == key)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting API key: {str(e)}")
            raise

    async def get_user_keys(
        self,
        user_id: int,
        skip: int = 0,
        limit: int = 100
    ) -> List[ApiKey]:
        """Get user's API keys."""
        try:
            query = select(ApiKey).where(ApiKey.user_id == user_id)
            query = query.offset(skip).limit(limit)
            result = await self.session.execute(query)
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error getting user API keys: {str(e)}")
            raise

    async def update_last_used(self, key: str) -> Optional[ApiKey]:
        """Update API key last used timestamp."""
        try:
            result = await self.session.execute(
                update(ApiKey)
                .where(ApiKey.key == key)
                .values(last_used_at=datetime.utcnow())
                .returning(ApiKey)
            )
            await self.session.commit()
            return result.scalar_one_or_none()
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error updating API key last used: {str(e)}")
            raise

class SecurityAuditRepository:
    def __init__(self, session: Session):
        self.session = session

    async def create(self, audit_data: Dict[str, Any]) -> SecurityAudit:
        """Create new security audit entry."""
        try:
            audit = SecurityAudit(**audit_data)
            self.session.add(audit)
            await self.session.commit()
            await self.session.refresh(audit)
            return audit
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error creating security audit: {str(e)}")
            raise

    async def get_user_audits(
        self,
        user_id: int,
        skip: int = 0,
        limit: int = 100,
        security_level: Optional[str] = None
    ) -> List[SecurityAudit]:
        """Get user's security audit entries."""
        try:
            query = select(SecurityAudit).where(SecurityAudit.user_id == user_id)
            if security_level:
                query = query.where(SecurityAudit.security_level == security_level)
            query = query.order_by(SecurityAudit.timestamp.desc()).offset(skip).limit(limit)
            result = await self.session.execute(query)
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error getting user security audits: {str(e)}")
            raise 