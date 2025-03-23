import os
import sys
from pathlib import Path
import logging
from alembic.config import Config
from alembic import command
from .connection import init_db
from ..models.config import DatabaseConfig
from ..config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_migrations():
    """Run database migrations."""
    try:
        # Get the directory containing this file
        current_dir = Path(__file__).parent
        migrations_dir = current_dir / "migrations"

        # Create Alembic configuration
        alembic_cfg = Config(str(migrations_dir / "alembic.ini"))
        alembic_cfg.set_main_option("script_location", str(migrations_dir))

        # Run migrations
        command.upgrade(alembic_cfg, "head")
        logger.info("Database migrations completed successfully")
    except Exception as e:
        logger.error(f"Error running migrations: {str(e)}")
        raise

def init_database():
    """Initialize database and run migrations."""
    try:
        # Initialize database connection
        db_config = DatabaseConfig(**settings.DATABASE_CONFIG)
        db = init_db(db_config)

        # Create database directory if using SQLite
        if db_config.type == "sqlite":
            db_dir = Path(db_config.database).parent
            db_dir.mkdir(parents=True, exist_ok=True)

        # Run migrations
        run_migrations()
        logger.info("Database initialization completed successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

if __name__ == "__main__":
    init_database() 