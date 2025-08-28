# core/error_handling.py

import logging
import traceback
from typing import Dict, Any, Optional
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .logging_config import get_logger, log_with_context

logger = get_logger(__name__)

class RAGException(Exception):
    """Base exception for RAG API errors."""
    
    def __init__(self, message: str, error_code: str = None, 
                 status_code: int = 500, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "INTERNAL_ERROR"
        self.status_code = status_code
        self.details = details or {}

class ValidationError(RAGException):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, "VALIDATION_ERROR", 400, details)

class RateLimitError(RAGException):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, "RATE_LIMIT_EXCEEDED", 429)

class ServiceUnavailableError(RAGException):
    """Raised when external services are unavailable."""
    
    def __init__(self, service: str, message: str = None):
        message = message or f"{service} service is unavailable"
        super().__init__(message, "SERVICE_UNAVAILABLE", 503, {"service": service})

class TenantNotFoundError(RAGException):
    """Raised when tenant is not found."""
    
    def __init__(self, tenant_id: str):
        super().__init__(f"Tenant '{tenant_id}' not found", "TENANT_NOT_FOUND", 404, {"tenant_id": tenant_id})

def create_error_response(exception: RAGException, request_id: str = None) -> Dict[str, Any]:
    """Create a standardized error response."""
    response = {
        "error": {
            "code": exception.error_code,
            "message": exception.message,
            "status_code": exception.status_code
        }
    }
    
    if exception.details:
        response["error"]["details"] = exception.details
    
    if request_id:
        response["request_id"] = request_id
    
    return response

async def rag_exception_handler(request: Request, exc: RAGException) -> JSONResponse:
    """Handle RAG-specific exceptions."""
    request_id = getattr(request.state, 'request_id', None)
    
    log_with_context(
        logger, logging.ERROR,
        f"RAG Exception: {exc.message}",
        request_id=request_id,
        error_code=exc.error_code,
        status_code=exc.status_code,
        details=exc.details
    )
    
    response_data = create_error_response(exc, request_id)
    return JSONResponse(
        status_code=exc.status_code,
        content=response_data
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle FastAPI validation exceptions."""
    request_id = getattr(request.state, 'request_id', None)
    
    error_details = []
    for error in exc.errors():
        error_details.append({
            "field": " -> ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    log_with_context(
        logger, logging.WARNING,
        f"Validation error: {len(error_details)} field(s) failed validation",
        request_id=request_id,
        error_details=error_details
    )
    
    response_data = {
        "error": {
            "code": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "status_code": 422,
            "details": error_details
        }
    }
    
    if request_id:
        response_data["request_id"] = request_id
    
    return JSONResponse(
        status_code=422,
        content=response_data
    )

async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle HTTP exceptions."""
    request_id = getattr(request.state, 'request_id', None)
    
    log_with_context(
        logger, logging.WARNING,
        f"HTTP Exception: {exc.detail}",
        request_id=request_id,
        status_code=exc.status_code
    )
    
    response_data = {
        "error": {
            "code": "HTTP_ERROR",
            "message": exc.detail,
            "status_code": exc.status_code
        }
    }
    
    if request_id:
        response_data["request_id"] = request_id
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response_data
    )

async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unhandled exceptions."""
    request_id = getattr(request.state, 'request_id', None)
    
    # Log the full exception with traceback
    log_with_context(
        logger, logging.ERROR,
        f"Unhandled exception: {str(exc)}",
        request_id=request_id,
        exception_type=type(exc).__name__,
        traceback=traceback.format_exc()
    )
    
    response_data = {
        "error": {
            "code": "INTERNAL_ERROR",
            "message": "An unexpected error occurred",
            "status_code": 500
        }
    }
    
    if request_id:
        response_data["request_id"] = request_id
    
    return JSONResponse(
        status_code=500,
        content=response_data
    )

def log_service_error(service: str, operation: str, error: Exception, 
                     tenant_id: str = None, session_id: str = None, 
                     request_id: str = None) -> None:
    """Log service-related errors with context."""
    log_with_context(
        logger, logging.ERROR,
        f"{service} service error during {operation}: {str(error)}",
        tenant_id=tenant_id,
        session_id=session_id,
        request_id=request_id,
        service=service,
        operation=operation,
        error_type=type(error).__name__
    )

def log_api_request(request: Request, tenant_id: str = None, session_id: str = None) -> None:
    """Log API request details."""
    request_id = getattr(request.state, 'request_id', None)
    
    log_with_context(
        logger, logging.INFO,
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
        logger, log_level,
        f"API Response: {status_code} ({response_time:.3f}s)",
        tenant_id=tenant_id,
        session_id=session_id,
        request_id=request_id,
        status_code=status_code,
        response_time=response_time
    )
