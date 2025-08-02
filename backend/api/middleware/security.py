"""
Security Headers Middleware
Adds security headers to responses
"""

import logging
from fastapi import Request
from fastapi.responses import Response
import structlog

logger = structlog.get_logger(__name__)


class SecurityHeadersMiddleware:
    """Security headers middleware"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Create a custom send function to modify headers
        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                # Add security headers
                headers = message.get("headers", [])
                headers.extend([
                    (b"X-Content-Type-Options", b"nosniff"),
                    (b"X-Frame-Options", b"DENY"),
                    (b"X-XSS-Protection", b"1; mode=block"),
                    (b"Referrer-Policy", b"strict-origin-when-cross-origin"),
                    (b"Content-Security-Policy", b"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';"),
                ])
                message["headers"] = headers
            
            await send(message)
        
        await self.app(scope, receive, send_with_headers)