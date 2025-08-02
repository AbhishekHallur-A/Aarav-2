"""
Rate Limiting Middleware
Handles API rate limiting using Redis
"""

import time
import logging
from typing import Dict, Optional
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import structlog

from utils.config import settings

logger = structlog.get_logger(__name__)


class RateLimitMiddleware:
    """Rate limiting middleware"""
    
    def __init__(self, app):
        self.app = app
        self.rate_limits = {}  # In-memory rate limit storage
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        
        # Get client IP
        client_ip = self._get_client_ip(request)
        
        # Check rate limit
        if not await self._check_rate_limit(client_ip):
            response = JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": "Too many requests. Please try again later."
                }
            )
            await response(scope, receive, send)
            return
        
        await self.app(scope, receive, send)
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    async def _check_rate_limit(self, client_ip: str) -> bool:
        """Check if client is within rate limits"""
        try:
            current_time = int(time.time())
            minute_key = f"rate_limit:{client_ip}:{current_time // 60}"
            
            # Get current count
            current_count = self.rate_limits.get(minute_key, 0)
            
            # Check if limit exceeded
            if current_count >= settings.RATE_LIMIT_PER_MINUTE:
                return False
            
            # Increment count
            self.rate_limits[minute_key] = current_count + 1
            
            # Clean up old entries (older than 1 minute)
            self._cleanup_old_entries(current_time)
            
            return True
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True  # Allow request on error
    
    def _cleanup_old_entries(self, current_time: int):
        """Clean up old rate limit entries"""
        try:
            current_minute = current_time // 60
            keys_to_remove = []
            
            for key in self.rate_limits.keys():
                try:
                    # Extract timestamp from key
                    parts = key.split(":")
                    if len(parts) >= 3:
                        timestamp = int(parts[2])
                        if timestamp < current_minute - 1:  # Older than 1 minute
                            keys_to_remove.append(key)
                except (ValueError, IndexError):
                    keys_to_remove.append(key)
            
            # Remove old entries
            for key in keys_to_remove:
                del self.rate_limits[key]
                
        except Exception as e:
            logger.error(f"Rate limit cleanup failed: {e}")