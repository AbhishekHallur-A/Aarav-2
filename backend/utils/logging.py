"""
Logging Configuration for AstraFind
Sets up structured logging with proper formatting and output
"""

import logging
import sys
from typing import Optional
from datetime import datetime

import structlog
from structlog.stdlib import LoggerFactory
from structlog.processors import (
    TimeStamper,
    JSONRenderer,
    format_exc_info,
    add_log_level,
    StackInfoRenderer
)

from .config import settings

def setup_logging(
    log_level: Optional[str] = None,
    json_output: bool = True,
    include_timestamp: bool = True
) -> None:
    """
    Setup structured logging for the application
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: Whether to output logs in JSON format
        include_timestamp: Whether to include timestamps in logs
    """
    
    # Use provided log level or default from settings
    level = log_level or settings.LOG_LEVEL
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )
    
    # Configure structlog
    processors = [
        # Add log level
        add_log_level,
        
        # Add stack info
        StackInfoRenderer(),
        
        # Add exception info
        format_exc_info,
    ]
    
    # Add timestamp if requested
    if include_timestamp:
        processors.insert(0, TimeStamper(fmt="iso"))
    
    # Add JSON renderer if requested
    if json_output:
        processors.append(JSONRenderer())
    else:
        # Use human-readable format
        processors.append(
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.exception_formatter
            )
        )
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


class RequestLogger:
    """Specialized logger for HTTP requests"""
    
    def __init__(self, name: str = "request"):
        self.logger = get_logger(name)
    
    def log_request(
        self,
        method: str,
        url: str,
        client_ip: str,
        user_agent: str,
        content_length: Optional[int] = None,
        **kwargs
    ):
        """Log incoming request"""
        self.logger.info(
            "HTTP request received",
            method=method,
            url=url,
            client_ip=client_ip,
            user_agent=user_agent,
            content_length=content_length,
            **kwargs
        )
    
    def log_response(
        self,
        method: str,
        url: str,
        status_code: int,
        process_time: float,
        content_length: Optional[int] = None,
        **kwargs
    ):
        """Log response"""
        self.logger.info(
            "HTTP response sent",
            method=method,
            url=url,
            status_code=status_code,
            process_time=process_time,
            content_length=content_length,
            **kwargs
        )
    
    def log_error(
        self,
        method: str,
        url: str,
        error: str,
        process_time: float,
        **kwargs
    ):
        """Log request error"""
        self.logger.error(
            "HTTP request failed",
            method=method,
            url=url,
            error=error,
            process_time=process_time,
            **kwargs
        )


class DatabaseLogger:
    """Specialized logger for database operations"""
    
    def __init__(self, name: str = "database"):
        self.logger = get_logger(name)
    
    def log_query(
        self,
        query: str,
        params: Optional[dict] = None,
        execution_time: Optional[float] = None,
        **kwargs
    ):
        """Log database query"""
        self.logger.debug(
            "Database query executed",
            query=query,
            params=params,
            execution_time=execution_time,
            **kwargs
        )
    
    def log_connection(self, status: str, **kwargs):
        """Log database connection event"""
        self.logger.info(
            "Database connection",
            status=status,
            **kwargs
        )
    
    def log_error(self, error: str, **kwargs):
        """Log database error"""
        self.logger.error(
            "Database error",
            error=error,
            **kwargs
        )


class SearchLogger:
    """Specialized logger for search operations"""
    
    def __init__(self, name: str = "search"):
        self.logger = get_logger(name)
    
    def log_search(
        self,
        query: str,
        results_count: int,
        search_time: float,
        user_id: Optional[str] = None,
        **kwargs
    ):
        """Log search operation"""
        self.logger.info(
            "Search performed",
            query=query,
            results_count=results_count,
            search_time=search_time,
            user_id=user_id,
            **kwargs
        )
    
    def log_indexing(
        self,
        document_id: str,
        url: str,
        status: str,
        **kwargs
    ):
        """Log document indexing"""
        self.logger.info(
            "Document indexed",
            document_id=document_id,
            url=url,
            status=status,
            **kwargs
        )
    
    def log_crawling(
        self,
        url: str,
        status: str,
        response_time: Optional[float] = None,
        **kwargs
    ):
        """Log crawling operation"""
        self.logger.info(
            "URL crawled",
            url=url,
            status=status,
            response_time=response_time,
            **kwargs
        )


class SecurityLogger:
    """Specialized logger for security events"""
    
    def __init__(self, name: str = "security"):
        self.logger = get_logger(name)
    
    def log_auth_success(
        self,
        user_id: str,
        method: str,
        client_ip: str,
        **kwargs
    ):
        """Log successful authentication"""
        self.logger.info(
            "Authentication successful",
            user_id=user_id,
            method=method,
            client_ip=client_ip,
            **kwargs
        )
    
    def log_auth_failure(
        self,
        username: str,
        method: str,
        client_ip: str,
        reason: str,
        **kwargs
    ):
        """Log failed authentication"""
        self.logger.warning(
            "Authentication failed",
            username=username,
            method=method,
            client_ip=client_ip,
            reason=reason,
            **kwargs
        )
    
    def log_rate_limit(
        self,
        client_ip: str,
        user_agent: str,
        **kwargs
    ):
        """Log rate limit exceeded"""
        self.logger.warning(
            "Rate limit exceeded",
            client_ip=client_ip,
            user_agent=user_agent,
            **kwargs
        )
    
    def log_suspicious_activity(
        self,
        activity_type: str,
        client_ip: str,
        details: str,
        **kwargs
    ):
        """Log suspicious activity"""
        self.logger.warning(
            "Suspicious activity detected",
            activity_type=activity_type,
            client_ip=client_ip,
            details=details,
            **kwargs
        )


# Create default loggers
request_logger = RequestLogger()
database_logger = DatabaseLogger()
search_logger = SearchLogger()
security_logger = SecurityLogger()


def log_performance(operation: str, duration: float, **kwargs):
    """Log performance metrics"""
    logger = get_logger("performance")
    logger.info(
        "Performance metric",
        operation=operation,
        duration=duration,
        **kwargs
    )


def log_health_check(service: str, status: str, **kwargs):
    """Log health check results"""
    logger = get_logger("health")
    logger.info(
        "Health check",
        service=service,
        status=status,
        **kwargs
    )


def log_startup(component: str, **kwargs):
    """Log component startup"""
    logger = get_logger("startup")
    logger.info(
        "Component started",
        component=component,
        **kwargs
    )


def log_shutdown(component: str, **kwargs):
    """Log component shutdown"""
    logger = get_logger("shutdown")
    logger.info(
        "Component stopped",
        component=component,
        **kwargs
    )