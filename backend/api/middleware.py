"""
Middleware for AstraFind API
Handles rate limiting, security headers, and request logging
"""

import time
import hashlib
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
import structlog

from ..utils.config import settings

logger = structlog.get_logger(__name__)

class RateLimitMiddleware:
    """Rate limiting middleware"""
    
    def __init__(self):
        self.requests: Dict[str, List[float]] = {}
        self.cleanup_interval = 60  # Clean up old requests every 60 seconds
        self.last_cleanup = time.time()
    
    async def __call__(self, request: Request, call_next):
        # Clean up old requests periodically
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            await self._cleanup_old_requests(current_time)
            self.last_cleanup = current_time
        
        # Get client identifier
        client_id = await self._get_client_id(request)
        
        # Check rate limit
        if not await self._check_rate_limit(client_id, current_time):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": "Too many requests. Please try again later.",
                    "retry_after": 60
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(settings.RATE_LIMIT_PER_MINUTE)
        response.headers["X-RateLimit-Remaining"] = str(
            await self._get_remaining_requests(client_id)
        )
        response.headers["X-RateLimit-Reset"] = str(
            int(current_time + 60)
        )
        
        return response
    
    async def _get_client_id(self, request: Request) -> str:
        """Get unique client identifier"""
        # Use IP address as primary identifier
        client_ip = request.client.host if request.client else "unknown"
        
        # Add user agent for additional uniqueness
        user_agent = request.headers.get("user-agent", "")
        
        # Create hash for privacy
        identifier = f"{client_ip}:{user_agent}"
        return hashlib.md5(identifier.encode()).hexdigest()
    
    async def _check_rate_limit(self, client_id: str, current_time: float) -> bool:
        """Check if client has exceeded rate limit"""
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Remove requests older than 1 minute
        cutoff_time = current_time - 60
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > cutoff_time
        ]
        
        # Check if limit exceeded
        if len(self.requests[client_id]) >= settings.RATE_LIMIT_PER_MINUTE:
            return False
        
        # Add current request
        self.requests[client_id].append(current_time)
        return True
    
    async def _get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for client"""
        if client_id not in self.requests:
            return settings.RATE_LIMIT_PER_MINUTE
        
        current_time = time.time()
        cutoff_time = current_time - 60
        
        # Count recent requests
        recent_requests = len([
            req_time for req_time in self.requests[client_id]
            if req_time > cutoff_time
        ])
        
        return max(0, settings.RATE_LIMIT_PER_MINUTE - recent_requests)
    
    async def _cleanup_old_requests(self, current_time: float):
        """Clean up old request records"""
        cutoff_time = current_time - 120  # Keep 2 minutes of history
        
        for client_id in list(self.requests.keys()):
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if req_time > cutoff_time
            ]
            
            # Remove empty client records
            if not self.requests[client_id]:
                del self.requests[client_id]


class SecurityHeadersMiddleware:
    """Add security headers to responses"""
    
    async def __call__(self, request: Request, call_next):
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # Content Security Policy
        csp_policy = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self' https:; "
            "media-src 'self' https:; "
            "object-src 'none'; "
            "base-uri 'self'; "
            "form-action 'self';"
        )
        response.headers["Content-Security-Policy"] = csp_policy
        
        # HSTS (only in production)
        if settings.ENVIRONMENT == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response


class RequestLoggingMiddleware:
    """Log all requests and responses"""
    
    async def __call__(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(
            "Request started",
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("user-agent", ""),
            content_length=request.headers.get("content-length", 0)
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                "Request completed",
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                process_time=process_time,
                content_length=response.headers.get("content-length", 0)
            )
            
            # Add processing time header
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log error
            logger.error(
                "Request failed",
                method=request.method,
                url=str(request.url),
                error=str(e),
                process_time=process_time,
                exc_info=True
            )
            
            # Re-raise the exception
            raise


class TrustedHostMiddleware:
    """Validate trusted hosts"""
    
    def __init__(self, allowed_hosts: List[str]):
        self.allowed_hosts = allowed_hosts
    
    async def __call__(self, request: Request, call_next):
        host = request.headers.get("host", "")
        
        # Remove port number if present
        if ":" in host:
            host = host.split(":")[0]
        
        # Check if host is allowed
        if host not in self.allowed_hosts and "*" not in self.allowed_hosts:
            logger.warning(f"Blocked request from untrusted host: {host}")
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Invalid host",
                    "message": "Request from untrusted host"
                }
            )
        
        return await call_next(request)


class CORSMiddleware:
    """CORS middleware for cross-origin requests"""
    
    def __init__(self, allow_origins: List[str], allow_credentials: bool = True):
        self.allow_origins = allow_origins
        self.allow_credentials = allow_credentials
    
    async def __call__(self, request: Request, call_next):
        response = await call_next(request)
        
        # Get origin from request
        origin = request.headers.get("origin")
        
        # Set CORS headers
        if origin and (origin in self.allow_origins or "*" in self.allow_origins):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = str(self.allow_credentials).lower()
        
        # Set other CORS headers
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
        
        return response


class GZipMiddleware:
    """Gzip compression middleware"""
    
    def __init__(self, minimum_size: int = 1000):
        self.minimum_size = minimum_size
    
    async def __call__(self, request: Request, call_next):
        response = await call_next(request)
        
        # Check if response should be compressed
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) >= self.minimum_size:
            # Check if client accepts gzip
            accept_encoding = request.headers.get("accept-encoding", "")
            if "gzip" in accept_encoding.lower():
                response.headers["Content-Encoding"] = "gzip"
                # Note: Actual compression would be handled by FastAPI/ASGI server
        
        return response