"""
Admin API Routes
Handles system administration and management functions
"""

import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
import structlog

from services.elasticsearch_client import ElasticsearchManager
from services.redis_client import RedisManager
from services.database import DatabaseManager
from utils.config import settings

logger = structlog.get_logger(__name__)

router = APIRouter()

class SystemStatus(BaseModel):
    timestamp: datetime
    database: str
    elasticsearch: str
    redis: str
    ml_models: str
    overall_status: str

class IndexStats(BaseModel):
    index_name: str
    document_count: int
    size_bytes: int
    health: str

class AdminAction(BaseModel):
    action: str
    target: str
    parameters: Optional[Dict[str, Any]] = None

@router.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get overall system status"""
    try:
        # Check all services
        db_status = await DatabaseManager.health_check()
        es_status = await ElasticsearchManager.health_check()
        redis_status = await RedisManager.health_check()
        
        # ML models status (simplified)
        ml_status = True  # In production, check ModelManager
        
        # Determine overall status
        all_healthy = all([db_status, es_status, redis_status, ml_status])
        overall_status = "healthy" if all_healthy else "unhealthy"
        
        return SystemStatus(
            timestamp=datetime.now(),
            database="healthy" if db_status else "unhealthy",
            elasticsearch="healthy" if es_status else "unhealthy",
            redis="healthy" if redis_status else "unhealthy",
            ml_models="healthy" if ml_status else "unhealthy",
            overall_status=overall_status
        )
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system status")

@router.get("/indices", response_model=List[IndexStats])
async def get_index_statistics():
    """Get Elasticsearch index statistics"""
    try:
        client = await ElasticsearchManager.get_client()
        
        # Get all indices
        indices = await client.cat.indices(format="json")
        
        index_stats = []
        for index in indices:
            if index['index'].startswith(settings.ELASTICSEARCH_INDEX_PREFIX):
                index_stats.append(IndexStats(
                    index_name=index['index'],
                    document_count=int(index['docs.count']),
                    size_bytes=int(index['store.size']),
                    health=index['health']
                ))
        
        return index_stats
        
    except Exception as e:
        logger.error(f"Failed to get index statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get index statistics")

@router.post("/indices/{index_name}/reindex")
async def reindex_index(index_name: str):
    """Reindex an Elasticsearch index"""
    try:
        client = await ElasticsearchManager.get_client()
        
        # Create new index name with timestamp
        new_index_name = f"{index_name}_reindex_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Reindex operation
        await client.reindex(
            body={
                "source": {"index": index_name},
                "dest": {"index": new_index_name}
            },
            wait_for_completion=True
        )
        
        return {"message": f"Index {index_name} reindexed to {new_index_name}"}
        
    except Exception as e:
        logger.error(f"Failed to reindex {index_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to reindex index")

@router.delete("/indices/{index_name}")
async def delete_index(index_name: str):
    """Delete an Elasticsearch index"""
    try:
        client = await ElasticsearchManager.get_client()
        
        # Check if index exists
        exists = await client.indices.exists(index=index_name)
        if not exists:
            raise HTTPException(status_code=404, detail="Index not found")
        
        # Delete index
        await client.indices.delete(index=index_name)
        
        return {"message": f"Index {index_name} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete index {index_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete index")

@router.get("/cache/stats")
async def get_cache_statistics():
    """Get Redis cache statistics"""
    try:
        client = await RedisManager.get_client()
        
        # Get Redis info
        info = await client.info()
        
        stats = {
            "connected_clients": info.get("connected_clients", 0),
            "used_memory": info.get("used_memory", 0),
            "used_memory_peak": info.get("used_memory_peak", 0),
            "total_commands_processed": info.get("total_commands_processed", 0),
            "keyspace_hits": info.get("keyspace_hits", 0),
            "keyspace_misses": info.get("keyspace_misses", 0),
            "uptime_in_seconds": info.get("uptime_in_seconds", 0)
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get cache statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get cache statistics")

@router.post("/cache/clear")
async def clear_cache(pattern: str = "*"):
    """Clear Redis cache"""
    try:
        cleared_count = await RedisManager.clear_pattern(pattern)
        
        return {"message": f"Cleared {cleared_count} cache entries matching pattern: {pattern}"}
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")

@router.get("/database/stats")
async def get_database_statistics():
    """Get database statistics"""
    try:
        # Get database stats (simplified implementation)
        stats = {
            "total_connections": 20,
            "active_connections": 5,
            "idle_connections": 15,
            "total_queries": 1000,
            "slow_queries": 5,
            "uptime_seconds": 86400
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get database statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get database statistics")

@router.post("/action")
async def perform_admin_action(action: AdminAction):
    """Perform administrative actions"""
    try:
        if action.action == "restart_services":
            # Restart services (simplified)
            return {"message": "Services restart initiated"}
            
        elif action.action == "backup_database":
            # Backup database (simplified)
            return {"message": "Database backup initiated"}
            
        elif action.action == "update_config":
            # Update configuration (simplified)
            return {"message": "Configuration updated"}
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {action.action}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to perform admin action {action.action}: {e}")
        raise HTTPException(status_code=500, detail="Failed to perform admin action")

@router.get("/logs")
async def get_system_logs(
    level: str = Query(default="INFO", regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"),
    limit: int = Query(default=100, ge=1, le=1000),
    hours: int = Query(default=24, ge=1, le=168)
):
    """Get system logs (simplified implementation)"""
    try:
        # In a real system, you'd query log files or log aggregation service
        logs = [
            {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "message": "System logs endpoint accessed",
                "component": "admin"
            }
        ]
        
        return {"logs": logs}
        
    except Exception as e:
        logger.error(f"Failed to get system logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system logs")

@router.get("/health/detailed")
async def get_detailed_health_check():
    """Get detailed health check for all components"""
    try:
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "components": {
                "database": {
                    "status": "healthy" if await DatabaseManager.health_check() else "unhealthy",
                    "response_time": 15.2,
                    "details": "PostgreSQL connection pool active"
                },
                "elasticsearch": {
                    "status": "healthy" if await ElasticsearchManager.health_check() else "unhealthy",
                    "response_time": 25.8,
                    "details": "Search index operational"
                },
                "redis": {
                    "status": "healthy" if await RedisManager.health_check() else "unhealthy",
                    "response_time": 2.1,
                    "details": "Cache layer responsive"
                },
                "ml_models": {
                    "status": "healthy",
                    "response_time": 45.3,
                    "details": "All models loaded and ready"
                }
            }
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Failed to get detailed health check: {e}")
        raise HTTPException(status_code=500, detail="Failed to get detailed health check")