from fastapi import APIRouter
from app.api.endpoints import videos, tasks, transcriptions, avatars, audio, sound_effects

api_router = APIRouter()

api_router.include_router(
    videos.router,
    prefix="/videos",
    tags=["videos"]
)

api_router.include_router(
    tasks.router,
    prefix="/tasks",
    tags=["tasks"]
)

api_router.include_router(
    transcriptions.router,
    prefix="/transcriptions",
    tags=["transcriptions"]
)

api_router.include_router(
    avatars.router,
    prefix="/avatars",
    tags=["avatars"]
)

api_router.include_router(
    audio.router,
    prefix="/audio",
    tags=["audio"]
)

api_router.include_router(
    sound_effects.router,
    prefix="/sound-effects",
    tags=["sound_effects"]
) 