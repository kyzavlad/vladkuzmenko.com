import os
import logging
from typing import Optional, Dict, List
from fastapi import FastAPI, HTTPException, Depends, Request, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.websockets import WebSocket, WebSocketDisconnect
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import asyncio
from pydantic import BaseModel, Field
import json
from pathlib import Path

from ..models.user import User, UserCreate, UserProfile
from ..models.auth import Token, TokenData
from ..models.video import VideoEditRequest, VideoEditResponse, VideoStatusResponse
from ..models.avatar import AvatarCreateRequest, AvatarResponse, AvatarGenerateRequest
from ..models.translation import VideoTranslationRequest
from ..integration.video_pipeline import VideoPipelineIntegrator, PipelineConfig
from ..integration.token_system import TokenSystemIntegrator, TokenConfig
from ..integration.security import SecurityIntegrator, SecurityConfig
from ..config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Clip Generation Service API",
    description="API for video editing, avatar creation, and translation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

# Create necessary directories
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.TEMP_DIR, exist_ok=True)

# Initialize integrators
video_pipeline = VideoPipelineIntegrator(
    model=None,  # Initialize with your model
    config=PipelineConfig()
)

token_system = TokenSystemIntegrator(TokenConfig())
security = SecurityIntegrator(SecurityConfig(encryption_key=settings.ENCRYPTION_KEY))

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)

manager = ConnectionManager()

# Authentication helpers
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = await User.get_by_username(token_data.username)
    if user is None:
        raise credentials_exception
    return user

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    # TODO: Implement rate limiting logic
    response = await call_next(request)
    return response

# Authentication endpoints
@app.post("/api/auth/register", response_model=UserProfile)
async def register_user(user: UserCreate):
    """Register a new user."""
    # Check if username exists
    if await User.get_by_username(user.username):
        raise HTTPException(status_code=400, detail="Username already registered")
    
    # Create new user
    hashed_password = get_password_hash(user.password)
    user_dict = user.dict()
    user_dict["hashed_password"] = hashed_password
    
    new_user = await User.create(**user_dict)
    return UserProfile.from_orm(new_user)

@app.post("/api/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login user and return access token."""
    user = await User.get_by_username(form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

@app.post("/api/auth/refresh-token", response_model=Token)
async def refresh_token(current_user: User = Depends(get_current_user)):
    """Refresh access token."""
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": current_user.username}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

@app.get("/api/auth/user-profile", response_model=UserProfile)
async def get_user_profile(current_user: User = Depends(get_current_user)):
    """Get current user profile."""
    return UserProfile.from_orm(current_user)

# Video editing endpoints
@app.post("/api/video/edit", response_model=VideoEditResponse)
async def edit_video(
    request: VideoEditRequest,
    video_file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Process video editing request."""
    try:
        # Save uploaded file
        file_path = Path(settings.UPLOAD_DIR) / f"{current_user.id}_{video_file.filename}"
        with open(file_path, "wb") as buffer:
            content = await video_file.read()
            buffer.write(content)
        
        # Process video
        job_id = await video_pipeline.process_video(
            str(file_path),
            job_id=f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        return VideoEditResponse(
            job_id=job_id,
            estimated_completion_time=300  # TODO: Calculate based on video length
        )
    
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/video/edit/status/{job_id}", response_model=VideoStatusResponse)
async def get_video_status(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get video processing status."""
    status = video_pipeline.get_job_status(job_id)
    if status["status"] == "not_found":
        raise HTTPException(status_code=404, detail="Job not found")
    return VideoStatusResponse(**status)

# Avatar endpoints
@app.post("/api/avatar/create", response_model=AvatarResponse)
async def create_avatar(
    request: AvatarCreateRequest,
    video_sample: UploadFile = File(...),
    additional_photos: List[UploadFile] = File([]),
    voice_sample: Optional[UploadFile] = File(None),
    current_user: User = Depends(get_current_user)
):
    """Create a new avatar."""
    try:
        # Save uploaded files
        video_path = Path(settings.UPLOAD_DIR) / f"avatar_{current_user.id}_video.mp4"
        with open(video_path, "wb") as buffer:
            content = await video_sample.read()
            buffer.write(content)
        
        photo_paths = []
        for photo in additional_photos:
            photo_path = Path(settings.UPLOAD_DIR) / f"avatar_{current_user.id}_{photo.filename}"
            with open(photo_path, "wb") as buffer:
                content = await photo.read()
                buffer.write(content)
            photo_paths.append(str(photo_path))
        
        voice_path = None
        if voice_sample:
            voice_path = Path(settings.UPLOAD_DIR) / f"avatar_{current_user.id}_voice.wav"
            with open(voice_path, "wb") as buffer:
                content = await voice_sample.read()
                buffer.write(content)
        
        # TODO: Implement avatar creation logic
        
        return AvatarResponse(
            avatar_id=f"avatar_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            processing_status="processing",
            preview_url=None
        )
    
    except Exception as e:
        logger.error(f"Error creating avatar: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/avatar/{avatar_id}", response_model=AvatarResponse)
async def get_avatar(
    avatar_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get avatar details."""
    # TODO: Implement avatar retrieval logic
    raise HTTPException(status_code=501, detail="Not implemented")

@app.post("/api/avatar/generate", response_model=VideoEditResponse)
async def generate_avatar_video(
    request: AvatarGenerateRequest,
    current_user: User = Depends(get_current_user)
):
    """Generate video using avatar."""
    # TODO: Implement avatar video generation logic
    raise HTTPException(status_code=501, detail="Not implemented")

# Video translation endpoints
@app.post("/api/video/translate", response_model=VideoEditResponse)
async def translate_video(
    request: VideoTranslationRequest,
    video_file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Process video translation request."""
    try:
        # Save uploaded file
        file_path = Path(settings.UPLOAD_DIR) / f"translate_{current_user.id}_{video_file.filename}"
        with open(file_path, "wb") as buffer:
            content = await video_file.read()
            buffer.write(content)
        
        # TODO: Implement video translation logic
        
        return VideoEditResponse(
            job_id=f"translate_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            estimated_completion_time=300  # TODO: Calculate based on video length
        )
    
    except Exception as e:
        logger.error(f"Error translating video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoints
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages
            await manager.send_message(f"Message received: {data}", client_id)
    except WebSocketDisconnect:
        manager.disconnect(client_id)

# Error handling
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 