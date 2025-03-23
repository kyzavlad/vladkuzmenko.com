import requests
from fastapi import Depends, HTTPException, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any
import logging

from app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Set up HTTP Bearer scheme for JWT authentication
security = HTTPBearer()

class AuthException(Exception):
    """Exception raised for authentication errors"""
    pass

async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """
    Verify JWT token and extract user ID
    
    This function authenticates the user by verifying their JWT token
    with the Authentication Service.
    
    Args:
        credentials: The HTTP Authorization credentials containing the JWT token
        
    Returns:
        str: The user ID extracted from the validated token
        
    Raises:
        HTTPException: If the token is invalid or the auth service is unavailable
    """
    token = credentials.credentials
    
    try:
        # Send token to Authentication Service for verification
        response = requests.post(
            f"{settings.AUTH_SERVICE_URL}/api/v1/auth/verify-token",
            json={"token": token}
        )
        
        # If the token is invalid, the auth service will return a non-200 status
        if response.status_code != 200:
            logger.warning(f"Invalid token: {response.status_code} - {response.text}")
            raise HTTPException(
                status_code=401, 
                detail="Invalid authentication token"
            )
        
        # Extract user ID from the response
        user_data = response.json()
        if "user_id" not in user_data:
            logger.error("Auth service returned data without user_id")
            raise HTTPException(
                status_code=500,
                detail="Authentication service returned invalid data"
            )
        
        return user_data["user_id"]
    
    except requests.RequestException as e:
        logger.error(f"Auth service connection error: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Authentication service unavailable: {str(e)}"
        )

async def verify_service_api_key(
    api_key: str = Header(..., alias="X-API-Key")
) -> bool:
    """
    Verify the service-to-service API key
    
    This function is used for authenticating internal service-to-service 
    communications.
    
    Args:
        api_key: The API key provided in the X-API-Key header
        
    Returns:
        bool: True if the API key is valid
        
    Raises:
        HTTPException: If the API key is invalid
    """
    if not settings.SERVICE_API_KEY:
        logger.error("SERVICE_API_KEY is not configured")
        raise HTTPException(
            status_code=500,
            detail="Service API key not configured"
        )
    
    if api_key != settings.SERVICE_API_KEY:
        logger.warning("Invalid service API key provided")
        raise HTTPException(
            status_code=401,
            detail="Invalid service API key"
        )
    
    return True

def get_user_info(user_id: str) -> Dict[str, Any]:
    """
    Get user information from the Authentication Service
    
    Args:
        user_id: The ID of the user to get information for
        
    Returns:
        Dict: Dictionary containing user information
        
    Raises:
        AuthException: If there was an error getting user information
    """
    try:
        # Send request to Authentication Service
        response = requests.get(
            f"{settings.AUTH_SERVICE_URL}/api/v1/users/{user_id}",
            headers={"X-API-Key": settings.SERVICE_API_KEY}
        )
        
        if response.status_code != 200:
            logger.error(f"Failed to get user info: {response.status_code} - {response.text}")
            raise AuthException(f"Failed to get user information: {response.text}")
        
        return response.json()
    
    except requests.RequestException as e:
        logger.error(f"Error connecting to auth service: {str(e)}")
        raise AuthException(f"Authentication service unavailable: {str(e)}")

def notify_video_processing_service(
    video_id: str, 
    transcription_id: str, 
    status: str,
    files: Optional[Dict[str, str]] = None
) -> bool:
    """
    Notify the Video Processing Service about a transcription update
    
    Args:
        video_id: The ID of the video in the Video Processing Service
        transcription_id: The ID of the transcription
        status: The new status of the transcription
        files: Optional dictionary of output file paths
        
    Returns:
        bool: True if notification was successful
        
    Raises:
        Exception: If there was an error notifying the Video Processing Service
    """
    try:
        data = {
            "transcription_id": transcription_id,
            "status": status
        }
        
        if files:
            data["files"] = files
        
        # Send notification to Video Processing Service
        response = requests.post(
            f"{settings.VIDEO_PROCESSING_SERVICE_URL}/api/v1/videos/{video_id}/transcription-update",
            json=data,
            headers={"X-API-Key": settings.SERVICE_API_KEY}
        )
        
        if response.status_code != 200:
            logger.error(f"Failed to notify Video Processing Service: {response.status_code} - {response.text}")
            return False
        
        return True
    
    except requests.RequestException as e:
        logger.error(f"Error connecting to Video Processing Service: {str(e)}")
        return False 