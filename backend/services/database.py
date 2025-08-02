"""
Database Manager Service
Handles PostgreSQL database connections and operations
"""

import asyncio
import logging
from typing import Optional
import asyncpg
from asyncpg import Pool

from utils.config import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database connection manager"""
    
    _pool: Optional[Pool] = None
    
    @classmethod
    async def initialize(cls) -> None:
        """Initialize database connection pool"""
        try:
            cls._pool = await asyncpg.create_pool(
                str(settings.DATABASE_URL),
                min_size=5,
                max_size=settings.DATABASE_POOL_SIZE,
                command_timeout=60
            )
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise
    
    @classmethod
    async def get_pool(cls) -> Pool:
        """Get database connection pool"""
        if cls._pool is None:
            await cls.initialize()
        return cls._pool
    
    @classmethod
    async def close(cls) -> None:
        """Close database connection pool"""
        if cls._pool:
            await cls._pool.close()
            cls._pool = None
            logger.info("Database connection pool closed")
    
    @classmethod
    async def health_check(cls) -> bool:
        """Check database health"""
        try:
            pool = await cls.get_pool()
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False