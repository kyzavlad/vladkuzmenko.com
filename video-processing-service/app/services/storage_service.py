import os
import logging
import shutil
from typing import Optional, Union, BinaryIO
import aiofiles
from pathlib import Path
from fastapi import UploadFile
from datetime import timedelta
import uuid

from app.core.config import settings

logger = logging.getLogger(__name__)


class StorageService:
    """
    Service to handle file storage operations.
    Supports local file system and S3 storage backends.
    """
    
    def __init__(self):
        self.storage_type = settings.STORAGE_TYPE
        self.local_storage_path = settings.LOCAL_STORAGE_PATH
        
        # Initialize S3 client if using S3 storage
        self._s3_client = None
        if self.storage_type == "s3":
            self._init_s3_client()
    
    def _init_s3_client(self):
        """Initialize S3 client for AWS operations"""
        try:
            import boto3
            from botocore.exceptions import NoCredentialsError
            
            self._s3_client = boto3.client(
                's3',
                region_name=settings.S3_REGION,
                aws_access_key_id=settings.S3_ACCESS_KEY,
                aws_secret_access_key=settings.S3_SECRET_KEY
            )
            self.s3_bucket = settings.S3_BUCKET_NAME
            logger.info(f"S3 client initialized for bucket: {self.s3_bucket}")
        except ImportError:
            logger.error("boto3 library not installed. S3 storage will not work.")
            raise ValueError("boto3 library required for S3 storage")
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise ValueError("AWS credentials not found")
        except Exception as e:
            logger.error(f"Error initializing S3 client: {str(e)}")
            raise
    
    async def save_upload(self, file: UploadFile, path: str) -> str:
        """
        Save an uploaded file to storage
        
        Args:
            file: The uploaded file
            path: Storage path relative to storage root
            
        Returns:
            The full storage path
        """
        if self.storage_type == "local":
            return await self._save_local(file, path)
        elif self.storage_type == "s3":
            return await self._save_s3(file, path)
        else:
            raise ValueError(f"Unsupported storage type: {self.storage_type}")
    
    async def _save_local(self, file: UploadFile, path: str) -> str:
        """Save file to local storage"""
        full_path = os.path.join(self.local_storage_path, path)
        
        # Create directory if it doesn't exist
        directory = os.path.dirname(full_path)
        os.makedirs(directory, exist_ok=True)
        
        try:
            # Reset file position to start
            await file.seek(0)
            
            # Write file
            async with aiofiles.open(full_path, 'wb') as out_file:
                content = await file.read()
                await out_file.write(content)
            
            logger.info(f"File saved to local storage: {full_path}")
            return path
        except Exception as e:
            logger.error(f"Error saving file to local storage: {str(e)}")
            raise
    
    async def _save_s3(self, file: UploadFile, path: str) -> str:
        """Save file to S3 storage"""
        if self._s3_client is None:
            self._init_s3_client()
        
        try:
            # Reset file position to start
            await file.seek(0)
            
            # Read file content
            content = await file.read()
            
            # Upload to S3
            self._s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=path,
                Body=content,
                ContentType=file.content_type
            )
            
            logger.info(f"File uploaded to S3: {path}")
            return path
        except Exception as e:
            logger.error(f"Error uploading file to S3: {str(e)}")
            raise
    
    async def get_file(self, path: str) -> Optional[Union[str, bytes]]:
        """
        Get a file from storage
        
        Args:
            path: Storage path relative to storage root
            
        Returns:
            File content or local path
        """
        if self.storage_type == "local":
            return self._get_local_path(path)
        elif self.storage_type == "s3":
            return await self._get_s3_content(path)
        else:
            raise ValueError(f"Unsupported storage type: {self.storage_type}")
    
    def _get_local_path(self, path: str) -> str:
        """Get full path to local file"""
        full_path = os.path.join(self.local_storage_path, path)
        if not os.path.exists(full_path):
            logger.warning(f"File not found in local storage: {full_path}")
            raise FileNotFoundError(f"File not found: {path}")
        return full_path
    
    async def _get_s3_content(self, path: str) -> bytes:
        """Get file content from S3"""
        if self._s3_client is None:
            self._init_s3_client()
        
        try:
            response = self._s3_client.get_object(
                Bucket=self.s3_bucket,
                Key=path
            )
            return response['Body'].read()
        except Exception as e:
            logger.error(f"Error getting file from S3: {str(e)}")
            raise
    
    async def get_presigned_url(self, path: str, expires_in: int = 3600) -> str:
        """
        Get a presigned URL for file access
        
        Args:
            path: Storage path relative to storage root
            expires_in: URL expiry time in seconds (default: 1 hour)
            
        Returns:
            Presigned URL for file access
        """
        if self.storage_type == "local":
            return self._get_local_url(path)
        elif self.storage_type == "s3":
            return self._get_s3_presigned_url(path, expires_in)
        else:
            raise ValueError(f"Unsupported storage type: {self.storage_type}")
    
    def _get_local_url(self, path: str) -> str:
        """
        For local storage, we return a path relative to the API.
        In a real implementation, this would be a URL to an endpoint serving static files.
        """
        full_path = os.path.join(self.local_storage_path, path)
        if not os.path.exists(full_path):
            logger.warning(f"File not found in local storage: {full_path}")
            raise FileNotFoundError(f"File not found: {path}")
        
        # Return a URL for the local development server
        # In production, this would be replaced with a real URL
        return f"/api/v1/files/{path}"
    
    def _get_s3_presigned_url(self, path: str, expires_in: int) -> str:
        """Get presigned URL for S3 file"""
        if self._s3_client is None:
            self._init_s3_client()
        
        try:
            url = self._s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.s3_bucket,
                    'Key': path
                },
                ExpiresIn=expires_in
            )
            return url
        except Exception as e:
            logger.error(f"Error generating presigned URL: {str(e)}")
            raise
    
    async def delete_file(self, path: str) -> bool:
        """
        Delete a file from storage
        
        Args:
            path: Storage path relative to storage root
            
        Returns:
            True if file was deleted, False otherwise
        """
        if self.storage_type == "local":
            return self._delete_local(path)
        elif self.storage_type == "s3":
            return self._delete_s3(path)
        else:
            raise ValueError(f"Unsupported storage type: {self.storage_type}")
    
    def _delete_local(self, path: str) -> bool:
        """Delete file from local storage"""
        full_path = os.path.join(self.local_storage_path, path)
        try:
            if os.path.exists(full_path):
                os.remove(full_path)
                logger.info(f"File deleted from local storage: {full_path}")
                return True
            logger.warning(f"File not found for deletion in local storage: {full_path}")
            return False
        except Exception as e:
            logger.error(f"Error deleting file from local storage: {str(e)}")
            return False
    
    def _delete_s3(self, path: str) -> bool:
        """Delete file from S3 storage"""
        if self._s3_client is None:
            self._init_s3_client()
        
        try:
            self._s3_client.delete_object(
                Bucket=self.s3_bucket,
                Key=path
            )
            logger.info(f"File deleted from S3: {path}")
            return True
        except Exception as e:
            logger.error(f"Error deleting file from S3: {str(e)}")
            return False
    
    async def copy_file(self, source_path: str, dest_path: str) -> bool:
        """
        Copy a file within storage
        
        Args:
            source_path: Source storage path
            dest_path: Destination storage path
            
        Returns:
            True if file was copied, False otherwise
        """
        if self.storage_type == "local":
            return self._copy_local(source_path, dest_path)
        elif self.storage_type == "s3":
            return self._copy_s3(source_path, dest_path)
        else:
            raise ValueError(f"Unsupported storage type: {self.storage_type}")
    
    def _copy_local(self, source_path: str, dest_path: str) -> bool:
        """Copy file in local storage"""
        source_full_path = os.path.join(self.local_storage_path, source_path)
        dest_full_path = os.path.join(self.local_storage_path, dest_path)
        
        # Create destination directory if it doesn't exist
        dest_dir = os.path.dirname(dest_full_path)
        os.makedirs(dest_dir, exist_ok=True)
        
        try:
            if os.path.exists(source_full_path):
                shutil.copy2(source_full_path, dest_full_path)
                logger.info(f"File copied in local storage: {source_full_path} -> {dest_full_path}")
                return True
            logger.warning(f"Source file not found for copy in local storage: {source_full_path}")
            return False
        except Exception as e:
            logger.error(f"Error copying file in local storage: {str(e)}")
            return False
    
    def _copy_s3(self, source_path: str, dest_path: str) -> bool:
        """Copy file in S3 storage"""
        if self._s3_client is None:
            self._init_s3_client()
        
        try:
            self._s3_client.copy_object(
                Bucket=self.s3_bucket,
                CopySource={'Bucket': self.s3_bucket, 'Key': source_path},
                Key=dest_path
            )
            logger.info(f"File copied in S3: {source_path} -> {dest_path}")
            return True
        except Exception as e:
            logger.error(f"Error copying file in S3: {str(e)}")
            return False
    
    async def file_exists(self, path: str) -> bool:
        """
        Check if a file exists in storage
        
        Args:
            path: Storage path relative to storage root
            
        Returns:
            True if file exists, False otherwise
        """
        if self.storage_type == "local":
            return self._local_file_exists(path)
        elif self.storage_type == "s3":
            return self._s3_file_exists(path)
        else:
            raise ValueError(f"Unsupported storage type: {self.storage_type}")
    
    def _local_file_exists(self, path: str) -> bool:
        """Check if file exists in local storage"""
        full_path = os.path.join(self.local_storage_path, path)
        return os.path.exists(full_path)
    
    def _s3_file_exists(self, path: str) -> bool:
        """Check if file exists in S3 storage"""
        if self._s3_client is None:
            self._init_s3_client()
        
        try:
            self._s3_client.head_object(
                Bucket=self.s3_bucket,
                Key=path
            )
            return True
        except Exception:
            return False 