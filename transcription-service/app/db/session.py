from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from app.core.config import settings
from app.models.transcription import Base

# Create SQLAlchemy engine
engine = create_engine(
    settings.SQLALCHEMY_DATABASE_URI,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialize the database by creating all tables"""
    Base.metadata.create_all(bind=engine)

def get_db() -> Session:
    """
    Dependency function to get a database session
    
    This function is used with FastAPI's dependency injection system
    to provide a database session for API endpoint functions.
    
    Yields:
        Session: A SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 