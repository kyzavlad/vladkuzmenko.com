import logging
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import jwt
from jwt.exceptions import InvalidTokenError, ExpiredSignatureError
import httpx

from app.core.config import settings
from app.services.sound_effects.sound_effects_library import SoundEffectsLibrary
from app.services.sound_effects.sound_effects_processor import SoundEffectsProcessor

logger = logging.getLogger(__name__)

# Setup OAuth2 with token URL
# In a real system, this would point to your auth service endpoint
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)


async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """
    Validate and decode the JWT token to get the current user.
    This is used as a dependency in protected API endpoints.
    
    If authentication is disabled in development mode and no token is provided,
    a simulated user is returned.
    
    Args:
        token: JWT token from the Authorization header
        
    Returns:
        Decoded token payload with user information
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    # Check if auth is enabled
    if not settings.AUTH_ENABLED:
        logger.warning("Authentication is disabled! Using test user.")
        return {
            "sub": "test-user-123",
            "email": "test@example.com",
            "name": "Test User",
            "scopes": ["video:read", "video:write"]
        }
    
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check authentication method
    if settings.AUTH_METHOD == "jwt":
        # Verify and decode JWT token locally
        return await _verify_jwt_token(token)
    elif settings.AUTH_METHOD == "auth_service":
        # Verify token by calling auth service
        return await _verify_token_with_auth_service(token)
    else:
        logger.error(f"Unsupported authentication method: {settings.AUTH_METHOD}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


async def _verify_jwt_token(token: str) -> Dict[str, Any]:
    """
    Verify JWT token locally using the public key.

    Args:
        token: JWT token to verify
        
    Returns:
        Decoded token payload
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_PUBLIC_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            audience=settings.JWT_AUDIENCE,
            options={"verify_exp": True}
        )
        return payload
    except ExpiredSignatureError:
        logger.warning("Token has expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except InvalidTokenError as e:
        logger.warning(f"Invalid token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Error verifying token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def _verify_token_with_auth_service(token: str) -> Dict[str, Any]:
    """
    Verify token by calling the authentication service.
    
    Args:
        token: Token to verify
        
    Returns:
        User information from auth service
        
    Raises:
        HTTPException: If token is invalid or auth service request fails
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                settings.AUTH_SERVICE_VERIFY_URL,
                json={"token": token},
                headers={"Content-Type": "application/json"},
                timeout=5.0
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            else:
                logger.error(f"Auth service error: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Error verifying token with auth service"
                )
    except httpx.RequestError as e:
        logger.error(f"Error connecting to auth service: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error connecting to authentication service"
        )


async def get_current_active_user(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Get the current active user. This adds an additional check to verify
    that the user account is active.
    
    Args:
        current_user: User information from token
        
    Returns:
        User information if account is active
        
    Raises:
        HTTPException: If user account is inactive
    """
    # Check if user is active (can be expanded based on your user model)
    if current_user.get("disabled", False):
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user 

# Sound Effects Services
_sound_effects_library_instance = None
_sound_effects_processor_instance = None

def get_sound_effects_library() -> SoundEffectsLibrary:
    """
    Create or return an existing instance of the SoundEffectsLibrary.
    
    Returns:
        SoundEffectsLibrary: An instance of the sound effects library
    """
    global _sound_effects_library_instance
    if _sound_effects_library_instance is None:
        logger.info("Initializing Sound Effects Library")
        _sound_effects_library_instance = SoundEffectsLibrary(config=settings.sound_effects_library)
    return _sound_effects_library_instance

def get_sound_effects_processor() -> SoundEffectsProcessor:
    """
    Create or return an existing instance of the SoundEffectsProcessor.
    
    Returns:
        SoundEffectsProcessor: An instance of the sound effects processor
    """
    global _sound_effects_processor_instance
    if _sound_effects_processor_instance is None:
        logger.info("Initializing Sound Effects Processor")
        _sound_effects_processor_instance = SoundEffectsProcessor(config=settings.sound_effects_processor)
    return _sound_effects_processor_instance 