# core/logging_config.py

import logging
import logging.config
import os
import sys
from typing import Dict, Any
import json
from datetime import datetime
from fastapi import Request

def setup_logging(environment: str = "development") -> None:
    """
    Set up centralized logging configuration.
    
    Args:
        environment: "development", "testing", or "production"
    """
    
    # Prevent multiple basicConfig calls
    if logging.getLogger().handlers:
        return
    
    if environment == "production":
        setup_production_logging()
    elif environment == "testing":
        setup_testing_logging()
    else:
        setup_development_logging()

def setup_development_logging() -> None:
    """Set up logging for development environment."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def setup_testing_logging() -> None:
    """Set up logging for testing environment."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def setup_production_logging() -> None:
    """Set up structured JSON logging for production environment."""
    
    class JSONFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno
            }
            
            # Add exception info if present
            if record.exc_info:
                log_entry["exception"] = self.formatException(record.exc_info)
            
            # Add extra fields if present
            if hasattr(record, 'tenant_id'):
                log_entry["tenant_id"] = record.tenant_id
            if hasattr(record, 'session_id'):
                log_entry["session_id"] = record.session_id
            if hasattr(record, 'request_id'):
                log_entry["request_id"] = record.request_id
            
            return json.dumps(log_entry)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",  # JSON formatter will handle the actual format
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Apply JSON formatter to root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.setFormatter(JSONFormatter())

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)

def log_with_context(logger: logging.Logger, level: int, message: str, 
                    tenant_id: str = None, session_id: str = None, 
                    request_id: str = None, **kwargs) -> None:
    """
    Log a message with additional context.
    
    Args:
        logger: The logger instance
        level: Log level (logging.INFO, etc.)
        message: Log message
        tenant_id: Tenant ID for context
        session_id: Session ID for context
        request_id: Request ID for correlation
        **kwargs: Additional context fields
    """
    extra = {}
    if tenant_id:
        extra['tenant_id'] = tenant_id
    if session_id:
        extra['session_id'] = session_id
    if request_id:
        extra['request_id'] = request_id
    extra.update(kwargs)
    
    logger.log(level, message, extra=extra)

def log_api_request(request: Request, tenant_id: str = None, session_id: str = None) -> None:
    """Log API request details."""
    request_id = getattr(request.state, 'request_id', None)
    
    log_with_context(
        get_logger(__name__), logging.INFO,
        f"API Request: {request.method} {request.url.path}",
        tenant_id=tenant_id,
        session_id=session_id,
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        client_ip=request.client.host if request.client else None
    )

def log_api_response(request: Request, status_code: int, response_time: float,
                    tenant_id: str = None, session_id: str = None) -> None:
    """Log API response details."""
    request_id = getattr(request.state, 'request_id', None)
    
    log_level = logging.INFO if status_code < 400 else logging.WARNING
    
    log_with_context(
        get_logger(__name__), log_level,
        f"API Response: {status_code} ({response_time:.3f}s)",
        tenant_id=tenant_id,
        session_id=session_id,
        request_id=request_id,
        status_code=status_code,
        response_time=response_time
    )

# Initialize logging based on environment
def initialize_logging(environment: str = None):
    """Initialize logging based on environment variables."""
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development")
    setup_logging(environment)
