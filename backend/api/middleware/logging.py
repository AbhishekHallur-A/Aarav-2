"""
Request Logging Middleware
Logs API requests and responses
"""

import time
import logging
from fastapi import Request
import structlog

logger = structlog.get_logger(__name__)


class RequestLoggingMiddleware:
    """Request logging middleware"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        start_time = time.time()
        
        # Create a custom send function to log response
        async def send_with_logging(message):
            if message["type"] == "http.response.start":
                # Log request completion
                process_time = time.time() - start_time
                status_code = message.get("status", 200)
                
                logger.info(
                    "Request processed",
                    method=scope.get("method", "UNKNOWN"),
                    path=scope.get("path", "/"),
                    status_code=status_code,
                    process_time=process_time
                )
            
            await send(message)
        
        # Log request start
        logger.info(
            "Request started",
            method=scope.get("method", "UNKNOWN"),
            path=scope.get("path", "/"),
            client_ip=scope.get("client", ("unknown", 0))[0]
        )
        
        await self.app(scope, receive, send_with_logging)