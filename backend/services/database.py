"""
Database Manager for AstraFind
Handles PostgreSQL database connections and operations
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import MetaData
import structlog

from ..utils.config import settings

logger = structlog.get_logger(__name__)

# SQLAlchemy setup
Base = declarative_base()
metadata = MetaData()

class DatabaseManager:
    """Manages database connections and operations"""
    
    _engine: Optional[AsyncSession] = None
    _session_factory: Optional[async_sessionmaker] = None
    _pool: Optional[asyncpg.Pool] = None
    _initialized: bool = False
    
    @classmethod
    async def initialize(cls) -> None:
        """Initialize database connections"""
        if cls._initialized:
            return
            
        try:
            # Create async engine
            database_url = str(settings.DATABASE_URL).replace('postgresql://', 'postgresql+asyncpg://')
            cls._engine = create_async_engine(
                database_url,
                pool_size=settings.DATABASE_POOL_SIZE,
                max_overflow=settings.DATABASE_MAX_OVERFLOW,
                echo=settings.DEBUG,
                pool_pre_ping=True
            )
            
            # Create session factory
            cls._session_factory = async_sessionmaker(
                cls._engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create asyncpg pool for raw SQL operations
            cls._pool = await asyncpg.create_pool(
                host=settings.DATABASE_HOST,
                port=settings.DATABASE_PORT,
                user=settings.DATABASE_USER,
                password=settings.DATABASE_PASSWORD,
                database=settings.DATABASE_NAME,
                min_size=5,
                max_size=settings.DATABASE_POOL_SIZE
            )
            
            # Test connection
            async with cls._engine.begin() as conn:
                await conn.run_sync(lambda sync_conn: sync_conn.execute("SELECT 1"))
            
            cls._initialized = True
            logger.info("Database connection initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    @classmethod
    async def close(cls) -> None:
        """Close database connections"""
        if cls._engine:
            await cls._engine.dispose()
        if cls._pool:
            await cls._pool.close()
        cls._initialized = False
        logger.info("Database connections closed")
    
    @classmethod
    @asynccontextmanager
    async def get_session(cls):
        """Get database session context manager"""
        if not cls._initialized:
            await cls.initialize()
        
        async with cls._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    @classmethod
    async def get_pool(cls) -> asyncpg.Pool:
        """Get asyncpg connection pool"""
        if not cls._initialized:
            await cls.initialize()
        return cls._pool
    
    @classmethod
    async def health_check(cls) -> bool:
        """Check database health"""
        try:
            if not cls._initialized:
                return False
            
            async with cls._engine.begin() as conn:
                await conn.run_sync(lambda sync_conn: sync_conn.execute("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    @classmethod
    async def execute_query(cls, query: str, *args, **kwargs) -> Any:
        """Execute raw SQL query"""
        pool = await cls.get_pool()
        async with pool.acquire() as conn:
            return await conn.fetch(query, *args, **kwargs)
    
    @classmethod
    async def execute_command(cls, command: str, *args, **kwargs) -> str:
        """Execute SQL command (INSERT, UPDATE, DELETE)"""
        pool = await cls.get_pool()
        async with pool.acquire() as conn:
            return await conn.execute(command, *args, **kwargs)
    
    @classmethod
    async def create_tables(cls) -> None:
        """Create database tables"""
        if not cls._initialized:
            await cls.initialize()
        
        async with cls._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created")
    
    @classmethod
    async def drop_tables(cls) -> None:
        """Drop all database tables"""
        if not cls._initialized:
            await cls.initialize()
        
        async with cls._engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.info("Database tables dropped")