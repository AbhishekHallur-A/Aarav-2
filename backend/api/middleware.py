import time
import asyncio
import json
import hashlib
from typing import Dict, Optional, Set, Tuple, Callable
from datetime import datetime, timedelta
from ipaddress import ip_address, ip_network
import re

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import structlog
import redis.asyncio as redis
from ..utils.config import settings

logger = structlog.get_logger(__name__)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        for header, value in self.security_headers.items():
            response.headers[header] = value
        
        # Add server header obfuscation
        response.headers["Server"] = "AstraFind/1.0"
        
        return response

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware with Redis backend"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.redis_client: Optional[redis.Redis] = None
        self.fallback_cache: Dict[str, Dict] = {}
        
        # Rate limiting rules
        self.rate_limits = {
            'default': {'requests': 100, 'window': 60},  # 100 requests per minute
            'search': {'requests': 50, 'window': 60},    # 50 searches per minute
            'upload': {'requests': 10, 'window': 60},    # 10 uploads per minute
            'auth': {'requests': 5, 'window': 300},      # 5 auth attempts per 5 minutes
        }
        
        # Exempt IPs (for internal services)
        self.exempt_ips = set(getattr(settings, 'RATE_LIMIT_EXEMPT_IPS', []))
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Initialize Redis client if not done
        if self.redis_client is None:
            try:
                self.redis_client = redis.from_url(settings.REDIS_URL)
                await self.redis_client.ping()
            except Exception as e:
                logger.warning(f"Redis connection failed, using fallback cache: {e}")
        
        # Check if IP is exempt
        client_ip = self._get_client_ip(request)
        if self._is_exempt_ip(client_ip):
            return await call_next(request)
        
        # Determine rate limit rule
        rule_name = self._get_rate_limit_rule(request)
        rule = self.rate_limits.get(rule_name, self.rate_limits['default'])
        
        # Check rate limit
        is_allowed, retry_after = await self._check_rate_limit(
            client_ip, rule_name, rule['requests'], rule['window']
        )
        
        if not is_allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after": retry_after,
                    "limit": rule['requests'],
                    "window": rule['window']
                },
                headers={"Retry-After": str(retry_after)}
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = await self._get_remaining_requests(client_ip, rule_name, rule['requests'], rule['window'])
        response.headers["X-RateLimit-Limit"] = str(rule['requests'])
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + rule['window'])
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to client IP
        return request.client.host if request.client else "unknown"
    
    def _is_exempt_ip(self, ip: str) -> bool:
        """Check if IP is exempt from rate limiting"""
        try:
            client_ip = ip_address(ip)
            for exempt_ip in self.exempt_ips:
                if '/' in exempt_ip:  # CIDR notation
                    if client_ip in ip_network(exempt_ip, strict=False):
                        return True
                else:  # Single IP
                    if client_ip == ip_address(exempt_ip):
                        return True
        except Exception as e:
            logger.warning(f"Error checking exempt IP {ip}: {e}")
        
        return False
    
    def _get_rate_limit_rule(self, request: Request) -> str:
        """Determine which rate limit rule to apply"""
        path = request.url.path
        
        if path.startswith("/api/v1/search"):
            return "search"
        elif path.startswith("/api/v1/upload"):
            return "upload"
        elif path.startswith("/api/v1/auth"):
            return "auth"
        else:
            return "default"
    
    async def _check_rate_limit(self, client_ip: str, rule_name: str, limit: int, window: int) -> Tuple[bool, int]:
        """Check if request is within rate limit"""
        key = f"rate_limit:{client_ip}:{rule_name}"
        current_time = int(time.time())
        
        try:
            if self.redis_client:
                # Use Redis for rate limiting
                pipe = self.redis_client.pipeline()
                pipe.zremrangebyscore(key, 0, current_time - window)
                pipe.zcard(key)
                pipe.zadd(key, {str(current_time): current_time})
                pipe.expire(key, window)
                results = await pipe.execute()
                
                current_requests = results[1]
                
                if current_requests >= limit:
                    # Calculate retry after
                    oldest_request = await self.redis_client.zrange(key, 0, 0, withscores=True)
                    if oldest_request:
                        retry_after = int(oldest_request[0][1]) + window - current_time
                        return False, max(retry_after, 1)
                
                return True, 0
            else:
                # Fallback to in-memory cache
                return self._check_rate_limit_fallback(client_ip, rule_name, limit, window)
                
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # Allow request on error
            return True, 0
    
    def _check_rate_limit_fallback(self, client_ip: str, rule_name: str, limit: int, window: int) -> Tuple[bool, int]:
        """Fallback rate limiting using in-memory cache"""
        key = f"{client_ip}:{rule_name}"
        current_time = time.time()
        
        if key not in self.fallback_cache:
            self.fallback_cache[key] = []
        
        # Remove old requests
        self.fallback_cache[key] = [
            req_time for req_time in self.fallback_cache[key]
            if current_time - req_time < window
        ]
        
        if len(self.fallback_cache[key]) >= limit:
            retry_after = int(window - (current_time - self.fallback_cache[key][0]))
            return False, max(retry_after, 1)
        
        self.fallback_cache[key].append(current_time)
        return True, 0
    
    async def _get_remaining_requests(self, client_ip: str, rule_name: str, limit: int, window: int) -> int:
        """Get remaining requests for the current window"""
        key = f"rate_limit:{client_ip}:{rule_name}"
        current_time = int(time.time())
        
        try:
            if self.redis_client:
                await self.redis_client.zremrangebyscore(key, 0, current_time - window)
                current_requests = await self.redis_client.zcard(key)
                return max(0, limit - current_requests)
            else:
                # Fallback calculation
                cache_key = f"{client_ip}:{rule_name}"
                if cache_key in self.fallback_cache:
                    current_requests = len([
                        req_time for req_time in self.fallback_cache[cache_key]
                        if time.time() - req_time < window
                    ])
                    return max(0, limit - current_requests)
                return limit
        except Exception:
            return limit

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all HTTP requests and responses"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.sensitive_headers = {
            'authorization', 'cookie', 'x-api-key', 'x-auth-token'
        }
        self.sensitive_paths = {
            '/api/v1/auth/login', '/api/v1/auth/register', '/api/v1/auth/reset-password'
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Log request
        request_log = self._create_request_log(request)
        logger.info("HTTP request started", **request_log)
        
        # Process request
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log response
            response_log = self._create_response_log(request, response, process_time)
            
            if response.status_code >= 400:
                logger.warning("HTTP request completed with error", **response_log)
            else:
                logger.info("HTTP request completed", **response_log)
            
            # Add timing header
            response.headers["X-Process-Time"] = f"{process_time:.4f}"
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            error_log = {
                **request_log,
                "status_code": 500,
                "error": str(e),
                "process_time": process_time
            }
            logger.error("HTTP request failed", **error_log)
            raise
    
    def _create_request_log(self, request: Request) -> Dict:
        """Create request log entry"""
        client_ip = request.headers.get("X-Forwarded-For", 
                                       request.headers.get("X-Real-IP", 
                                                          request.client.host if request.client else "unknown"))
        
        # Filter sensitive headers
        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in self.sensitive_headers
        }
        
        log_data = {
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "client_ip": client_ip,
            "user_agent": request.headers.get("User-Agent", ""),
            "content_type": request.headers.get("Content-Type", ""),
            "content_length": request.headers.get("Content-Length", 0),
        }
        
        # Add headers for non-sensitive paths
        if request.url.path not in self.sensitive_paths:
            log_data["headers"] = headers
        
        return log_data
    
    def _create_response_log(self, request: Request, response: Response, process_time: float) -> Dict:
        """Create response log entry"""
        return {
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "process_time": process_time,
            "response_size": response.headers.get("Content-Length", 0)
        }

class IPWhitelistMiddleware(BaseHTTPMiddleware):
    """IP whitelist middleware for admin endpoints"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.admin_paths = {'/api/v1/admin', '/metrics', '/health/detailed'}
        self.whitelist = set(getattr(settings, 'ADMIN_IP_WHITELIST', ['127.0.0.1', '::1']))
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check if this is an admin path
        is_admin_path = any(request.url.path.startswith(path) for path in self.admin_paths)
        
        if is_admin_path:
            client_ip = self._get_client_ip(request)
            
            if not self._is_whitelisted(client_ip):
                logger.warning(f"Unauthorized admin access attempt from {client_ip}")
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"error": "Access denied"}
                )
        
        return await call_next(request)
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _is_whitelisted(self, ip: str) -> bool:
        """Check if IP is whitelisted"""
        try:
            client_ip = ip_address(ip)
            for allowed_ip in self.whitelist:
                if '/' in allowed_ip:  # CIDR notation
                    if client_ip in ip_network(allowed_ip, strict=False):
                        return True
                else:  # Single IP
                    if client_ip == ip_address(allowed_ip):
                        return True
        except Exception as e:
            logger.warning(f"Error checking whitelist for IP {ip}: {e}")
        
        return False

class ContentValidationMiddleware(BaseHTTPMiddleware):
    """Validate request content and detect potential attacks"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        
        # SQL injection patterns
        self.sql_injection_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC)\b)",
            r"(\b(UNION|JOIN)\b.*\b(SELECT)\b)",
            r"(--|#|/\*)",
            r"(\b(OR|AND)\b.*[=<>].*\b(OR|AND)\b)",
        ]
        
        # XSS patterns
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>.*?</iframe>",
        ]
        
        # Command injection patterns
        self.command_injection_patterns = [
            r"[;&|`$()]",
            r"\b(cat|ls|pwd|whoami|id|uname)\b",
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Only validate POST/PUT requests with content
        if request.method in ["POST", "PUT", "PATCH"] and "content-type" in request.headers:
            content_type = request.headers["content-type"].lower()
            
            if "application/json" in content_type:
                body = await request.body()
                if body:
                    try:
                        # Parse JSON to validate
                        json_data = json.loads(body)
                        
                        # Check for malicious content
                        if self._contains_malicious_content(json_data):
                            logger.warning(f"Malicious content detected from {request.client.host}")
                            return JSONResponse(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                content={"error": "Invalid content detected"}
                            )
                        
                        # Store parsed body for later use
                        request.state.json_body = json_data
                        
                    except json.JSONDecodeError:
                        return JSONResponse(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            content={"error": "Invalid JSON"}
                        )
            
            elif "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
                # Let FastAPI handle form parsing, but we could add checks here
                pass
        
        return await call_next(request)
    
    def _contains_malicious_content(self, data) -> bool:
        """Check if data contains malicious patterns"""
        if isinstance(data, dict):
            for key, value in data.items():
                if self._contains_malicious_content(value):
                    return True
        elif isinstance(data, list):
            for item in data:
                if self._contains_malicious_content(item):
                    return True
        elif isinstance(data, str):
            return self._check_string_patterns(data)
        
        return False
    
    def _check_string_patterns(self, text: str) -> bool:
        """Check string for malicious patterns"""
        text_lower = text.lower()
        
        # Check SQL injection
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        # Check XSS
        for pattern in self.xss_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        # Check command injection
        for pattern in self.command_injection_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        return False

class CompressionMiddleware(BaseHTTPMiddleware):
    """Custom compression middleware with better control"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.minimum_size = 500  # Only compress responses larger than 500 bytes
        self.compressible_types = {
            'application/json',
            'application/xml',
            'text/html',
            'text/plain',
            'text/css',
            'text/javascript',
            'application/javascript',
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Check if client accepts gzip
        accept_encoding = request.headers.get("Accept-Encoding", "")
        if "gzip" not in accept_encoding:
            return response
        
        # Check content type
        content_type = response.headers.get("Content-Type", "").split(";")[0]
        if content_type not in self.compressible_types:
            return response
        
        # Check content length
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) < self.minimum_size:
            return response
        
        # Response is already handled by GZipMiddleware in FastAPI
        return response

class APIVersionMiddleware(BaseHTTPMiddleware):
    """Handle API versioning"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.current_version = "1.0"
        self.supported_versions = {"1.0", "1.1"}
        self.deprecated_versions = set()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Add API version to response headers
        response = await call_next(request)
        response.headers["API-Version"] = self.current_version
        response.headers["Supported-Versions"] = ",".join(self.supported_versions)
        
        if self.deprecated_versions:
            response.headers["Deprecated-Versions"] = ",".join(self.deprecated_versions)
        
        return response

class HealthCheckBypassMiddleware(BaseHTTPMiddleware):
    """Bypass heavy middleware for health checks"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.health_paths = {"/health", "/ping", "/status"}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # For health check endpoints, set a flag to bypass heavy processing
        if request.url.path in self.health_paths:
            request.state.is_health_check = True
        
        return await call_next(request)