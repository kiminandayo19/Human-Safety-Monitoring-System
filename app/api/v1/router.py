"""Aggregates all v1 API routers."""

from fastapi import APIRouter

from app.api.v1.endpoints import detections, health, monitoring

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(detections.router)
api_router.include_router(monitoring.router)
