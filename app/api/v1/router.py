"""Aggregates all v1 API routers."""

from fastapi import APIRouter

from app.api.v1.endpoints import detect, detections, health, monitoring, track

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(detections.router)
api_router.include_router(monitoring.router)
api_router.include_router(detect.router)
api_router.include_router(track.router)
