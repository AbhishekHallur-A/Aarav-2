"""
Redis Client Service
Handles Redis connections and caching operations
"""

import logging
import json
from typing import Optional, Any, Dict
import redis.asyncio as redis

from utils.config import settings

logger = logging.getLogger(__name__)


class RedisManager:
    """Redis connection manager"""
    
    _client: Optional[redis.Redis] = None
    
    @classmethod
    async def initialize(cls) -> None:
        """Initialize Redis client"""
        try:
            cls._client = redis.Redis.from_url(
                str(settings.REDIS_URL),
                decode_responses=True,
                max_connections=settings.REDIS_POOL_SIZE
            )
            # Test connection
            await cls._client.ping()
            logger.info("Redis client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
            raise
    
    @classmethod
    async def get_client(cls) -> redis.Redis:
        """Get Redis client"""
        if cls._client is None:
            await cls.initialize()
        return cls._client
    
    @classmethod
    async def close(cls) -> None:
        """Close Redis client"""
        if cls._client:
            await cls._client.close()
            cls._client = None
            logger.info("Redis client closed")
    
    @classmethod
    async def health_check(cls) -> bool:
        """Check Redis health"""
        try:
            client = await cls.get_client()
            await client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    @classmethod
    async def set_cache(cls, key: str, value: Any, expire: int = 3600) -> bool:
        """Set cache value"""
        try:
            client = await cls.get_client()
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            await client.setex(key, expire, value)
            return True
        except Exception as e:
            logger.error(f"Failed to set cache: {e}")
            return False
    
    @classmethod
    async def get_cache(cls, key: str) -> Optional[Any]:
        """Get cache value"""
        try:
            client = await cls.get_client()
            value = await client.get(key)
            if value:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return None
        except Exception as e:
            logger.error(f"Failed to get cache: {e}")
            return None
    
    @classmethod
    async def delete_cache(cls, key: str) -> bool:
        """Delete cache value"""
        try:
            client = await cls.get_client()
            await client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Failed to delete cache: {e}")
            return False