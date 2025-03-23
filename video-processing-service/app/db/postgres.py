import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import text

from app.core.config import settings

logger = logging.getLogger(__name__)

# SQLAlchemy async engine and session factory
engine = create_async_engine(settings.DATABASE_URI, echo=False)
async_session = sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False,
    autocommit=False, 
    autoflush=False
)

# Base model class
Base = declarative_base()


async def init_db():
    """
    Initialize the database, create tables if they don't exist.
    Called during application startup.
    """
    try:
        # Create tables if they don't exist
        async with engine.begin() as conn:
            # Import all models that should create tables
            from app.models.video import Video, ProcessingTask
            
            # Create tables
            await conn.run_sync(Base.metadata.create_all)
        
        # Verify connection with a simple query
        async with async_session() as session:
            result = await session.execute(text("SELECT 1"))
            result.scalar_one()
            logger.info("Database connection established successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise


async def get_db():
    """
    Dependency for FastAPI endpoints to get a database session.
    Ensures proper session cleanup after request completion.
    """
    session = async_session()
    try:
        yield session
    finally:
        await session.close()


async def close_db_connection():
    """
    Close database connections on application shutdown.
    """
    await engine.dispose()
    logger.info("Database connections closed") 