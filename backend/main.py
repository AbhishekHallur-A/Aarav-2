#!/usr/bin/env python3
"""
AstraFind - AI-Driven Search Engine
Main FastAPI Application Entry Point
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
import structlog

from api.routes import api_router
from api.middleware import (
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    RequestLoggingMiddleware
)
from services.database import DatabaseManager
from services.elasticsearch_client import ElasticsearchManager
from services.redis_client import RedisManager
from services.ml.model_manager import ModelManager
from utils.config import settings
from utils.logging import setup_logging


# Setup structured logging
setup_logging()
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan management"""
    logger.info("Starting AstraFind application...")
    
    try:
        # Initialize database connections
        await DatabaseManager.initialize()
        logger.info("Database connection initialized")
        
        # Initialize Elasticsearch
        await ElasticsearchManager.initialize()
        logger.info("Elasticsearch connection initialized")
        
        # Initialize Redis
        await RedisManager.initialize()
        logger.info("Redis connection initialized")
        
        # Initialize ML models
        await ModelManager.initialize()
        logger.info("ML models loaded")
        
        # Create default indices if they don't exist
        await ElasticsearchManager.create_default_indices()
        logger.info("Elasticsearch indices verified")
        
        logger.info("AstraFind application started successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)
    
    finally:
        # Cleanup on shutdown
        logger.info("Shutting down AstraFind application...")
        await DatabaseManager.close()
        await ElasticsearchManager.close()
        await RedisManager.close()
        await ModelManager.cleanup()
        logger.info("Application shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="AstraFind API",
    description="AI-Driven Search Engine API",
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
    lifespan=lifespan
)

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Security Middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

app.add_middleware(SecurityHeadersMiddleware)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compression Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Rate Limiting Middleware
app.add_middleware(RateLimitMiddleware)

# Request Logging Middleware
app.add_middleware(RequestLoggingMiddleware)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Global HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "status_code": exc.status_code,
                "path": str(request.url.path)
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "status_code": 500,
                "path": str(request.url.path)
            }
        }
    )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to AstraFind AI-Driven Search Engine",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "metrics": "/metrics"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        db_status = await DatabaseManager.health_check()
        
        # Check Elasticsearch connection
        es_status = await ElasticsearchManager.health_check()
        
        # Check Redis connection
        redis_status = await RedisManager.health_check()
        
        # Check ML models
        ml_status = await ModelManager.health_check()
        
        all_healthy = all([db_status, es_status, redis_status, ml_status])
        
        return {
            "status": "healthy" if all_healthy else "unhealthy",
            "timestamp": asyncio.get_event_loop().time(),
            "services": {
                "database": "healthy" if db_status else "unhealthy",
                "elasticsearch": "healthy" if es_status else "unhealthy",
                "redis": "healthy" if redis_status else "unhealthy",
                "ml_models": "healthy" if ml_status else "unhealthy"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


# Include API routes
app.include_router(api_router, prefix="/api/v1")


if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        access_log=settings.DEBUG,
        workers=1 if settings.DEBUG else settings.WORKERS
    )