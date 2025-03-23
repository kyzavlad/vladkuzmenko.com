from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base
from motor.motor_asyncio import AsyncIOMotorClient
from redis import Redis
from typing import AsyncGenerator, Generator

from ..core.config import settings

# PostgreSQL
engine = create_engine(settings.SQLALCHEMY_DATABASE_URI, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# MongoDB
mongo_client = AsyncIOMotorClient(settings.MONGODB_URL)
mongo_db = mongo_client[settings.MONGODB_DB]

# Redis
redis_client = Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    password=settings.REDIS_PASSWORD,
    decode_responses=True
)

# Dependency
def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSession(engine) as session:
        try:
            yield session
        finally:
            await session.close()

# MongoDB collections
def get_mongo_collection(collection_name: str):
    return mongo_db[collection_name]

# Redis connection
def get_redis():
    return redis_client 