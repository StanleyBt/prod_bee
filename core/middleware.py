# core/middleware.py

import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .logging_config import get_logger, log_api_request, log_api_response
from .error_handling import log_service_error

logger = get_logger(__name__)

class RequestCorrelationMiddleware(BaseHTTPMiddleware):
    """Middleware for request correlation and logging."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Extract tenant and session from query params or headers
        tenant_id = request.query_params.get("tenant_id") or request.headers.get("X-Tenant-ID")
        session_id = request.query_params.get("session_id") or request.headers.get("X-Session-ID")
        
        # Store in request state for access throughout the request
        request.state.tenant_id = tenant_id
        request.state.session_id = session_id
        
        # Log request start
        log_api_request(request, tenant_id, session_id)
        
        # Record start time
        start_time = time.time()
        
        try:
            # Process the request
            response = await call_next(request)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Log response
            log_api_response(request, response.status_code, response_time, tenant_id, session_id)
            
            # Add correlation headers
            response.headers["X-Request-ID"] = request_id
            if tenant_id:
                response.headers["X-Tenant-ID"] = tenant_id
            
            return response
            
        except Exception as e:
            # Calculate response time for error cases
            response_time = time.time() - start_time
            
            # Log error
            log_service_error(
                "API", "request_processing", e,
                tenant_id=tenant_id,
                session_id=session_id,
                request_id=request_id
            )
            
            # Re-raise the exception for proper handling
            raise

class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting basic metrics."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.request_count = 0
        self.error_count = 0
        self.response_times = []
    
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        self.request_count += 1
        start_time = time.time()
        
        try:
            response = await call_next(request)
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            
            # Keep only last 1000 response times for memory efficiency
            if len(self.response_times) > 1000:
                self.response_times = self.response_times[-1000:]
            
            return response
            
        except Exception as e:
            self.error_count += 1
            response_time = time.time() - start_time
            raise
    
    def get_metrics(self) -> dict:
        """Get current metrics."""
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        return {
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": self.error_count / self.request_count if self.request_count > 0 else 0,
            "average_response_time": avg_response_time,
            "min_response_time": min(self.response_times) if self.response_times else 0,
            "max_response_time": max(self.response_times) if self.response_times else 0
        }
