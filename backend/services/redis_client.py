"""
Redis Manager for AstraFind
Handles Redis connections and caching operations
"""

import asyncio
import json
import pickle
from typing import Optional, Any, Dict, List
from datetime import timedelta

import redis.asyncio as redis
import structlog

from utils.config import settings

logger = structlog.get_logger(__name__)

class RedisManager:
    """Manages Redis connections and caching operations"""
    
    _client: Optional[redis.Redis] = None
    _initialized: bool = False
    
    @classmethod
    async def initialize(cls) -> None:
        """Initialize Redis connection"""
        if cls._initialized:
            return
            
        try:
            cls._client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                db=settings.REDIS_DB,
                decode_responses=False,  # Keep as bytes for pickle compatibility
                max_connections=settings.REDIS_POOL_SIZE
            )
            
            # Test connection
            await cls._client.ping()
            cls._initialized = True
            logger.info("Redis connection initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            raise
    
    @classmethod
    async def close(cls) -> None:
        """Close Redis connection"""
        if cls._client:
            await cls._client.close()
        cls._initialized = False
        logger.info("Redis connection closed")
    
    @classmethod
    async def get_client(cls) -> redis.Redis:
        """Get Redis client"""
        if not cls._initialized:
            await cls.initialize()
        return cls._client
    
    @classmethod
    async def health_check(cls) -> bool:
        """Check Redis health"""
        try:
            if not cls._initialized:
                return False
            
            client = await cls.get_client()
            await client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    @classmethod
    async def set_cache(cls, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """Set cache value"""
        try:
            client = await cls.get_client()
            
            # Serialize value
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value).encode('utf-8')
            else:
                serialized_value = pickle.dumps(value)
            
            await client.set(key, serialized_value, ex=expire)
            return True
        except Exception as e:
            logger.error(f"Failed to set cache key {key}: {e}")
            return False
    
    @classmethod
    async def get_cache(cls, key: str) -> Optional[Any]:
        """Get cache value"""
        try:
            client = await cls.get_client()
            value = await client.get(key)
            
            if value is None:
                return None
            
            # Try to deserialize as JSON first, then pickle
            try:
                return json.loads(value.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return pickle.loads(value)
                
        except Exception as e:
            logger.error(f"Failed to get cache key {key}: {e}")
            return None
    
    @classmethod
    async def delete_cache(cls, key: str) -> bool:
        """Delete cache key"""
        try:
            client = await cls.get_client()
            result = await client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Failed to delete cache key {key}: {e}")
            return False
    
    @classmethod
    async def exists_cache(cls, key: str) -> bool:
        """Check if cache key exists"""
        try:
            client = await cls.get_client()
            return await client.exists(key) > 0
        except Exception as e:
            logger.error(f"Failed to check cache key {key}: {e}")
            return False
    
    @classmethod
    async def set_hash(cls, key: str, mapping: Dict[str, Any], expire: Optional[int] = None) -> bool:
        """Set hash in Redis"""
        try:
            client = await cls.get_client()
            
            # Serialize values
            serialized_mapping = {}
            for field, value in mapping.items():
                if isinstance(value, (dict, list)):
                    serialized_mapping[field] = json.dumps(value).encode('utf-8')
                else:
                    serialized_mapping[field] = pickle.dumps(value)
            
            await client.hset(key, mapping=serialized_mapping)
            
            if expire:
                await client.expire(key, expire)
            
            return True
        except Exception as e:
            logger.error(f"Failed to set hash {key}: {e}")
            return False
    
    @classmethod
    async def get_hash(cls, key: str, fields: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """Get hash from Redis"""
        try:
            client = await cls.get_client()
            
            if fields:
                values = await client.hmget(key, fields)
                result = {}
                for field, value in zip(fields, values):
                    if value is not None:
                        try:
                            result[field] = json.loads(value.decode('utf-8'))
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            result[field] = pickle.loads(value)
                return result
            else:
                values = await client.hgetall(key)
                result = {}
                for field, value in values.items():
                    field = field.decode('utf-8')
                    try:
                        result[field] = json.loads(value.decode('utf-8'))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        result[field] = pickle.loads(value)
                return result
                
        except Exception as e:
            logger.error(f"Failed to get hash {key}: {e}")
            return None
    
    @classmethod
    async def increment_counter(cls, key: str, amount: int = 1, expire: Optional[int] = None) -> Optional[int]:
        """Increment counter in Redis"""
        try:
            client = await cls.get_client()
            result = await client.incrby(key, amount)
            
            if expire:
                await client.expire(key, expire)
            
            return result
        except Exception as e:
            logger.error(f"Failed to increment counter {key}: {e}")
            return None
    
    @classmethod
    async def get_counter(cls, key: str) -> Optional[int]:
        """Get counter value from Redis"""
        try:
            client = await cls.get_client()
            value = await client.get(key)
            return int(value) if value else 0
        except Exception as e:
            logger.error(f"Failed to get counter {key}: {e}")
            return None
    
    @classmethod
    async def add_to_set(cls, key: str, *values: str, expire: Optional[int] = None) -> bool:
        """Add values to Redis set"""
        try:
            client = await cls.get_client()
            await client.sadd(key, *values)
            
            if expire:
                await client.expire(key, expire)
            
            return True
        except Exception as e:
            logger.error(f"Failed to add to set {key}: {e}")
            return False
    
    @classmethod
    async def get_set_members(cls, key: str) -> List[str]:
        """Get all members of a Redis set"""
        try:
            client = await cls.get_client()
            members = await client.smembers(key)
            return [member.decode('utf-8') for member in members]
        except Exception as e:
            logger.error(f"Failed to get set members {key}: {e}")
            return []
    
    @classmethod
    async def add_to_list(cls, key: str, *values: str, expire: Optional[int] = None) -> bool:
        """Add values to Redis list"""
        try:
            client = await cls.get_client()
            await client.lpush(key, *values)
            
            if expire:
                await client.expire(key, expire)
            
            return True
        except Exception as e:
            logger.error(f"Failed to add to list {key}: {e}")
            return False
    
    @classmethod
    async def get_list_range(cls, key: str, start: int = 0, end: int = -1) -> List[str]:
        """Get range from Redis list"""
        try:
            client = await cls.get_client()
            values = await client.lrange(key, start, end)
            return [value.decode('utf-8') for value in values]
        except Exception as e:
            logger.error(f"Failed to get list range {key}: {e}")
            return []
    
    @classmethod
    async def clear_pattern(cls, pattern: str) -> int:
        """Clear keys matching pattern"""
        try:
            client = await cls.get_client()
            keys = await client.keys(pattern)
            if keys:
                return await client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Failed to clear pattern {pattern}: {e}")
            return 0