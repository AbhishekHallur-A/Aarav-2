"""
API Routes Module
Combines all route modules into a single router
"""

from fastapi import APIRouter

from .search import router as search_router
from .upload import router as upload_router
from .analytics import router as analytics_router
from .admin import router as admin_router
from .auth import router as auth_router

# Create main API router
api_router = APIRouter()

# Include all route modules
api_router.include_router(search_router, prefix="/search", tags=["search"])
api_router.include_router(upload_router, prefix="/upload", tags=["upload"])
api_router.include_router(analytics_router, prefix="/analytics", tags=["analytics"])
api_router.include_router(admin_router, prefix="/admin", tags=["admin"])
api_router.include_router(auth_router, prefix="/auth", tags=["authentication"])

__all__ = ["api_router"]