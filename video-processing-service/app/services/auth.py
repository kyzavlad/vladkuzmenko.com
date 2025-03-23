import requests
from fastapi import Depends, HTTPException, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any

from app.core.config import settings

security = HTTPBearer()

class AuthException(Exception):
    """Exception raised for authentication errors"""
    pass

async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """
    Verify JWT token from the Authorization header and return the user ID
    
    This function contacts the Authentication Service to validate the token
    and retrieve the user ID associated with it.
    
    Args:
        credentials: The HTTP Authorization credentials containing the JWT token
        
    Returns:
        str: The user ID extracted from the validated token
        
    Raises:
        HTTPException: If the token is invalid or the auth service is unavailable
    """
    token = credentials.credentials
    
    try:
        # Call the Authentication Service to validate the token
        response = requests.post(
            f"{settings.AUTH_SERVICE_URL}/api/v1/auth/verify-token",
            json={"token": token}
        )
        
        # If the token is invalid, the auth service will return a non-200 status
        if response.status_code != 200:
            raise HTTPException(
                status_code=401, 
                detail="Invalid authentication token"
            )
        
        # Extract user ID from the response
        user_data = response.json()
        if "user_id" not in user_data:
            raise HTTPException(
                status_code=500,
                detail="Authentication service returned invalid data"
            )
            
        return user_data["user_id"]
        
    except requests.RequestException as e:
        # Handle connection errors to the auth service
        raise HTTPException(
            status_code=503,
            detail=f"Authentication service unavailable: {str(e)}"
        )

async def verify_service_api_key(
    api_key: str = Header(..., alias="X-API-Key")
) -> bool:
    """
    Verify a service-to-service API key for internal communication
    
    This function is used for authenticating requests from other services
    in the microservice architecture.
    
    Args:
        api_key: The API key provided in the X-API-Key header
        
    Returns:
        bool: True if the API key is valid
        
    Raises:
        HTTPException: If the API key is invalid
    """
    if not settings.SERVICE_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Service API key not configured"
        )
        
    if api_key != settings.SERVICE_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid service API key"
        )
        
    return True

def get_user_info(user_id: str) -> Dict[str, Any]:
    """
    Retrieve user information from the Authentication Service
    
    Args:
        user_id: The ID of the user to retrieve information for
        
    Returns:
        Dict: User information including email, name, etc.
        
    Raises:
        AuthException: If the user information cannot be retrieved
    """
    try:
        # Call the Authentication Service to get user info
        response = requests.get(
            f"{settings.AUTH_SERVICE_URL}/api/v1/users/{user_id}",
            headers={"X-API-Key": settings.SERVICE_API_KEY}
        )
        
        if response.status_code != 200:
            raise AuthException(f"Failed to retrieve user info: {response.text}")
            
        return response.json()
        
    except requests.RequestException as e:
        raise AuthException(f"Authentication service unavailable: {str(e)}") 