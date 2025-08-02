"""
Structured Logging Configuration for AstraFind
"""

import logging
import sys
from typing import Any, Dict

import structlog
from structlog.stdlib import LoggerFactory

from .config import settings


def setup_logging() -> None:
    """Configure structured logging for the application"""
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.LOG_LEVEL),
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            # Add log level to event dict
            structlog.stdlib.add_log_level,
            # Add logger name to event dict
            structlog.stdlib.add_logger_name,
            # Add timestamp
            structlog.processors.TimeStamper(fmt="iso"),
            # Add filename and line number for debugging
            structlog.dev.set_exc_info if settings.DEBUG else structlog.processors.format_exc_info,
            # Format for console output in development
            structlog.dev.ConsoleRenderer(colors=True) if settings.DEBUG else structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance"""
    return structlog.get_logger(name)


class RequestLoggingFilter(logging.Filter):
    """Filter to add request context to logs"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add request context to log record"""
        # This will be populated by middleware
        record.request_id = getattr(record, 'request_id', None)
        record.user_id = getattr(record, 'user_id', None)
        record.ip_address = getattr(record, 'ip_address', None)
        return True


def bind_request_context(
    request_id: str,
    user_id: str = None,
    ip_address: str = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """Bind request context for structured logging"""
    context = {
        "request_id": request_id,
        "ip_address": ip_address,
        **kwargs
    }
    
    if user_id:
        context["user_id"] = user_id
    
    return context