import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Base configuration."""
    SECRET_KEY = os.getenv('SECRET_KEY', 'default-secret-key')
    DEBUG = os.getenv('DEBUG', 'False') == 'True'
    TESTING = False
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', './uploads')
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 500000000))  # 500MB default
    
    # OpenAI API Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # FFmpeg Configuration
    FFMPEG_PATH = os.getenv('FFMPEG_PATH', 'ffmpeg')
    FFPROBE_PATH = os.getenv('FFPROBE_PATH', 'ffprobe')


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True


# Configuration dictionary
config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig
}

def get_config():
    """Returns the current configuration based on the environment."""
    env = os.getenv('FLASK_ENV', 'development')
    return config_by_name[env] 