from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from contextlib import contextmanager
import logging
from ..models.config import DatabaseConfig

logger = logging.getLogger(__name__)

class Database:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine = self._create_engine()
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        self.Base = declarative_base()

    def _create_engine(self):
        """Create SQLAlchemy engine based on database configuration."""
        if self.config.type == "sqlite":
            database_url = f"sqlite:///{self.config.database}"
        else:
            database_url = (
                f"{self.config.type}://{self.config.username}:{self.config.password}"
                f"@{self.config.host}:{self.config.port}/{self.config.database}"
            )

        return create_engine(
            database_url,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_timeout=self.config.pool_timeout,
            pool_recycle=self.config.pool_recycle,
            echo=self.config.echo
        )

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {str(e)}")
            raise
        finally:
            session.close()

    def create_tables(self):
        """Create all database tables."""
        try:
            self.Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {str(e)}")
            raise

    def drop_tables(self):
        """Drop all database tables."""
        try:
            self.Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Error dropping database tables: {str(e)}")
            raise

# Global database instance
db = None

def init_db(config: DatabaseConfig):
    """Initialize global database instance."""
    global db
    db = Database(config)
    return db

def get_db() -> Database:
    """Get global database instance."""
    if db is None:
        raise RuntimeError("Database not initialized. Call init_db first.")
    return db 