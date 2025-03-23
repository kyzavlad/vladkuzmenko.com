"""
Settings Module

This module provides configuration settings for the Clip Generation Service.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Settings:
    """
    Configuration settings for the Clip Generation Service.
    
    This class handles loading settings from environment variables or a config file.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize settings.
        
        Args:
            config_file: Optional path to a configuration file
        """
        # Default settings
        self.defaults = {
            # Service settings
            "service_name": "clip-generation-service",
            "service_version": "1.0.0",
            "api_port": 8080,
            "debug": False,
            
            # Processing settings
            "output_dir": "output/clips",
            "temp_dir": "temp/clips",
            "ffmpeg_path": "ffmpeg",
            "ffprobe_path": "ffprobe",
            "default_format": "mp4",
            "default_quality": "high",
            
            # Queue settings
            "queue_type": "local",  # 'local', 'redis', 'rabbitmq', etc.
            "queue_host": "localhost",
            "queue_port": 6379,
            "queue_name": "clip-generation-tasks",
            
            # Worker settings
            "worker_count": 2,
            "max_concurrent_tasks": 5,
            "worker_timeout": 3600,  # 1 hour
            
            # Storage settings
            "storage_type": "local",  # 'local', 's3', 'gcs', etc.
            "storage_bucket": "clip-generation",
            "storage_path": "clips",
            
            # GPU settings
            "use_gpu": True,
            "gpu_memory_fraction": 0.5,
            "cuda_visible_devices": "",
            
            # Security settings
            "api_key_required": False,
            "allowed_origins": ["*"],
            "max_upload_size": 1024 * 1024 * 1024,  # 1 GB
            
            # Integration settings
            "video_processing_service_url": "http://localhost:8081",
            "notification_webhook_url": "",
            
            # Logging settings
            "log_level": "INFO",
            "log_file": "",
        }
        
        # Load settings from config file if provided
        self.settings = self.defaults.copy()
        if config_file:
            self._load_from_file(config_file)
        
        # Override with environment variables
        self._load_from_env()
        
        # Apply settings
        self._apply_settings()
        
        logger.info("Settings loaded")
    
    def _load_from_file(self, config_file: str) -> None:
        """
        Load settings from a JSON config file.
        
        Args:
            config_file: Path to the configuration file
        """
        if not os.path.exists(config_file):
            logger.warning(f"Config file not found: {config_file}")
            return
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Update settings
            self.settings.update(config)
            logger.info(f"Loaded settings from {config_file}")
            
        except Exception as e:
            logger.error(f"Error loading config file: {str(e)}")
    
    def _load_from_env(self) -> None:
        """
        Load settings from environment variables.
        """
        # Define mapping of environment variable names to settings keys
        env_mapping = {
            "CGS_SERVICE_NAME": "service_name",
            "CGS_SERVICE_VERSION": "service_version",
            "CGS_API_PORT": "api_port",
            "CGS_DEBUG": "debug",
            
            "CGS_OUTPUT_DIR": "output_dir",
            "CGS_TEMP_DIR": "temp_dir",
            "CGS_FFMPEG_PATH": "ffmpeg_path",
            "CGS_FFPROBE_PATH": "ffprobe_path",
            "CGS_DEFAULT_FORMAT": "default_format",
            "CGS_DEFAULT_QUALITY": "default_quality",
            
            "CGS_QUEUE_TYPE": "queue_type",
            "CGS_QUEUE_HOST": "queue_host",
            "CGS_QUEUE_PORT": "queue_port",
            "CGS_QUEUE_NAME": "queue_name",
            
            "CGS_WORKER_COUNT": "worker_count",
            "CGS_MAX_CONCURRENT_TASKS": "max_concurrent_tasks",
            "CGS_WORKER_TIMEOUT": "worker_timeout",
            
            "CGS_STORAGE_TYPE": "storage_type",
            "CGS_STORAGE_BUCKET": "storage_bucket",
            "CGS_STORAGE_PATH": "storage_path",
            
            "CGS_USE_GPU": "use_gpu",
            "CGS_GPU_MEMORY_FRACTION": "gpu_memory_fraction",
            "CGS_CUDA_VISIBLE_DEVICES": "cuda_visible_devices",
            
            "CGS_API_KEY_REQUIRED": "api_key_required",
            "CGS_ALLOWED_ORIGINS": "allowed_origins",
            "CGS_MAX_UPLOAD_SIZE": "max_upload_size",
            
            "CGS_VIDEO_PROCESSING_SERVICE_URL": "video_processing_service_url",
            "CGS_NOTIFICATION_WEBHOOK_URL": "notification_webhook_url",
            
            "CGS_LOG_LEVEL": "log_level",
            "CGS_LOG_FILE": "log_file",
        }
        
        # Process environment variables
        for env_var, setting_key in env_mapping.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # Convert boolean values
                if value.lower() in ["true", "yes", "1"]:
                    value = True
                elif value.lower() in ["false", "no", "0"]:
                    value = False
                # Convert integer values
                elif value.isdigit():
                    value = int(value)
                # Convert float values
                elif value.replace(".", "", 1).isdigit():
                    value = float(value)
                # Convert list values (comma-separated)
                elif "," in value and setting_key in ["allowed_origins"]:
                    value = [item.strip() for item in value.split(",")]
                
                # Update setting
                self.settings[setting_key] = value
                logger.debug(f"Setting {setting_key} from environment: {value}")
    
    def _apply_settings(self) -> None:
        """
        Apply settings after loading.
        """
        # Ensure directories exist
        os.makedirs(self.settings["output_dir"], exist_ok=True)
        os.makedirs(self.settings["temp_dir"], exist_ok=True)
        
        # Set up logging
        log_level = getattr(logging, self.settings["log_level"].upper(), logging.INFO)
        logging.getLogger().setLevel(log_level)
        
        # Set CUDA_VISIBLE_DEVICES for GPU settings
        if self.settings["use_gpu"] and self.settings["cuda_visible_devices"]:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.settings["cuda_visible_devices"]
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value.
        
        Args:
            key: Setting key
            default: Default value if setting doesn't exist
            
        Returns:
            Setting value or default
        """
        return self.settings.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """
        Allow dictionary-like access to settings.
        
        Args:
            key: Setting key
            
        Returns:
            Setting value
            
        Raises:
            KeyError: If setting doesn't exist
        """
        if key not in self.settings:
            raise KeyError(f"Setting not found: {key}")
        
        return self.settings[key]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert settings to a dictionary.
        
        Returns:
            Dictionary of settings
        """
        return self.settings.copy()


# Create a singleton instance
settings = Settings(
    config_file=os.environ.get("CGS_CONFIG_FILE")
) 