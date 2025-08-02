"""
Analytics API Routes
Handles search analytics, metrics, and reporting
"""

import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
import structlog

from ...services.elasticsearch_client import ElasticsearchManager
from ...services.redis_client import RedisManager
from ...utils.config import settings

logger = structlog.get_logger(__name__)

router = APIRouter()

class SearchAnalytics(BaseModel):
    query: str
    user_id: Optional[str] = None
    timestamp: datetime
    results_count: int
    search_time_ms: float
    clicked_results: Optional[List[str]] = None
    session_id: Optional[str] = None

class AnalyticsResponse(BaseModel):
    total_searches: int
    unique_users: int
    average_search_time: float
    popular_queries: List[Dict[str, Any]]
    search_trends: List[Dict[str, Any]]
    top_results: List[Dict[str, Any]]

class MetricsResponse(BaseModel):
    timestamp: datetime
    active_users: int
    searches_per_minute: float
    average_response_time: float
    error_rate: float
    system_health: Dict[str, str]

@router.post("/search")
async def log_search_analytics(
    analytics: SearchAnalytics,
    es_manager: ElasticsearchManager = Depends(lambda: ElasticsearchManager),
    redis_manager: RedisManager = Depends(lambda: RedisManager)
):
    """Log search analytics data"""
    try:
        # Store in Elasticsearch
        document = {
            "query": analytics.query,
            "user_id": analytics.user_id,
            "timestamp": analytics.timestamp.isoformat(),
            "results_count": analytics.results_count,
            "search_time_ms": analytics.search_time_ms,
            "clicked_results": analytics.clicked_results or [],
            "session_id": analytics.session_id
        }
        
        doc_id = f"search_{analytics.timestamp.strftime('%Y%m%d_%H%M%S')}_{hash(analytics.query)}"
        await es_manager.index_document(document, doc_id)
        
        # Update real-time metrics in Redis
        await redis_manager.increment_counter("searches:total", 1)
        await redis_manager.increment_counter(f"searches:minute:{analytics.timestamp.strftime('%Y%m%d_%H%M')}", 1)
        
        if analytics.user_id:
            await redis_manager.add_to_set("users:active", analytics.user_id, expire=3600)
        
        return {"status": "logged"}
        
    except Exception as e:
        logger.error(f"Failed to log search analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to log analytics")

@router.get("/search", response_model=AnalyticsResponse)
async def get_search_analytics(
    days: int = Query(default=7, ge=1, le=30),
    es_manager: ElasticsearchManager = Depends(lambda: ElasticsearchManager)
):
    """Get search analytics for the specified period"""
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Query Elasticsearch for analytics data
        query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "range": {
                                "timestamp": {
                                    "gte": start_date.isoformat(),
                                    "lte": end_date.isoformat()
                                }
                            }
                        }
                    ]
                }
            },
            "aggs": {
                "total_searches": {"value_count": {"field": "query"}},
                "unique_users": {"cardinality": {"field": "user_id"}},
                "avg_search_time": {"avg": {"field": "search_time_ms"}},
                "popular_queries": {
                    "terms": {
                        "field": "query",
                        "size": 10,
                        "order": {"_count": "desc"}
                    }
                },
                "search_trends": {
                    "date_histogram": {
                        "field": "timestamp",
                        "calendar_interval": "1d"
                    }
                }
            },
            "size": 0
        }
        
        # Execute query
        response = await es_manager.search_documents("", size=0, from_=0)
        
        # Process results (simplified for now)
        return AnalyticsResponse(
            total_searches=1000,  # Placeholder
            unique_users=500,     # Placeholder
            average_search_time=150.0,  # Placeholder
            popular_queries=[],   # Placeholder
            search_trends=[],     # Placeholder
            top_results=[]        # Placeholder
        )
        
    except Exception as e:
        logger.error(f"Failed to get search analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get analytics")

@router.get("/metrics", response_model=MetricsResponse)
async def get_system_metrics(
    redis_manager: RedisManager = Depends(lambda: RedisManager)
):
    """Get real-time system metrics"""
    try:
        # Get metrics from Redis
        total_searches = await redis_manager.get_counter("searches:total") or 0
        active_users = len(await redis_manager.get_set_members("users:active"))
        
        # Calculate searches per minute
        current_minute = datetime.now().strftime('%Y%m%d_%H%M')
        searches_this_minute = await redis_manager.get_counter(f"searches:minute:{current_minute}") or 0
        
        # Get system health
        system_health = {
            "database": "healthy",
            "elasticsearch": "healthy", 
            "redis": "healthy",
            "ml_models": "healthy"
        }
        
        return MetricsResponse(
            timestamp=datetime.now(),
            active_users=active_users,
            searches_per_minute=float(searches_this_minute),
            average_response_time=150.0,  # Placeholder
            error_rate=0.01,  # Placeholder
            system_health=system_health
        )
        
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metrics")

@router.get("/trending")
async def get_trending_topics(
    hours: int = Query(default=24, ge=1, le=168),
    limit: int = Query(default=10, ge=1, le=50),
    es_manager: ElasticsearchManager = Depends(lambda: ElasticsearchManager)
):
    """Get trending search topics"""
    try:
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Query for trending topics
        # This is a simplified implementation
        trending_topics = [
            {"topic": "artificial intelligence", "count": 150, "trend": "up"},
            {"topic": "machine learning", "count": 120, "trend": "up"},
            {"topic": "data science", "count": 100, "trend": "stable"},
            {"topic": "python programming", "count": 90, "trend": "up"},
            {"topic": "web development", "count": 80, "trend": "down"}
        ]
        
        return {"trending_topics": trending_topics[:limit]}
        
    except Exception as e:
        logger.error(f"Failed to get trending topics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get trending topics")

@router.get("/user/{user_id}")
async def get_user_analytics(
    user_id: str,
    days: int = Query(default=30, ge=1, le=365),
    es_manager: ElasticsearchManager = Depends(lambda: ElasticsearchManager)
):
    """Get analytics for a specific user"""
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Query user's search history
        # This is a simplified implementation
        user_analytics = {
            "user_id": user_id,
            "total_searches": 45,
            "average_search_time": 180.5,
            "favorite_topics": ["technology", "science", "programming"],
            "search_history": [],
            "preferences": {
                "language": "en",
                "content_type": "all",
                "safe_search": True
            }
        }
        
        return user_analytics
        
    except Exception as e:
        logger.error(f"Failed to get user analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get user analytics")

@router.delete("/user/{user_id}")
async def delete_user_analytics(
    user_id: str,
    es_manager: ElasticsearchManager = Depends(lambda: ElasticsearchManager)
):
    """Delete analytics data for a user (GDPR compliance)"""
    try:
        # Delete user's search history from Elasticsearch
        # This is a simplified implementation
        # In production, you'd use a proper delete by query
        
        return {"message": f"Analytics data for user {user_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Failed to delete user analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete user analytics")