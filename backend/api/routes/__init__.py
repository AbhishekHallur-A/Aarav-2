"""
API Routes Module
Combines all route modules into a single router
"""

from fastapi import APIRouter

from .search import router as search_router

# Create main API router
api_router = APIRouter()

# Include all route modules
api_router.include_router(search_router, prefix="/search", tags=["search"])

__all__ = ["api_router"]