from fastapi import APIRouter

from app.api.health import router as health_router
from app.api.models import router as models_router
from app.api.transcriptions import router as transcriptions_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(models_router)
api_router.include_router(transcriptions_router)
