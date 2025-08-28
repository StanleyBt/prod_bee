# tests/test_error_handling.py

import pytest
import logging
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.exceptions import RequestValidationError

from core.error_handling import (
    RAGException, ValidationError, RateLimitError, ServiceUnavailableError,
    TenantNotFoundError, create_error_response
)
from core.logging_config import setup_logging, get_logger, log_with_context


class TestRAGExceptions:
    """Test RAG-specific exceptions."""
    
    def test_rag_exception_creation(self):
        """Test creating RAG exceptions."""
        exc = RAGException("Test error", "TEST_ERROR", 400, {"field": "value"})
        assert exc.message == "Test error"
        assert exc.error_code == "TEST_ERROR"
        assert exc.status_code == 400
        assert exc.details == {"field": "value"}
    
    def test_validation_error(self):
        """Test ValidationError creation."""
        exc = ValidationError("Invalid input", {"field": "required"})
        assert exc.message == "Invalid input"
        assert exc.error_code == "VALIDATION_ERROR"
        assert exc.status_code == 400
        assert exc.details == {"field": "required"}
    
    def test_rate_limit_error(self):
        """Test RateLimitError creation."""
        exc = RateLimitError("Too many requests")
        assert exc.message == "Too many requests"
        assert exc.error_code == "RATE_LIMIT_EXCEEDED"
        assert exc.status_code == 429
    
    def test_service_unavailable_error(self):
        """Test ServiceUnavailableError creation."""
        exc = ServiceUnavailableError("Weaviate", "Connection failed")
        assert exc.message == "Connection failed"
        assert exc.error_code == "SERVICE_UNAVAILABLE"
        assert exc.status_code == 503
        assert exc.details == {"service": "Weaviate"}
    
    def test_tenant_not_found_error(self):
        """Test TenantNotFoundError creation."""
        exc = TenantNotFoundError("INVALID_TENANT")
        assert exc.message == "Tenant 'INVALID_TENANT' not found"
        assert exc.error_code == "TENANT_NOT_FOUND"
        assert exc.status_code == 404
        assert exc.details == {"tenant_id": "INVALID_TENANT"}


class TestErrorResponse:
    """Test error response creation."""
    
    def test_create_error_response(self):
        """Test creating standardized error responses."""
        exc = ValidationError("Invalid input", {"field": "required"})
        response = create_error_response(exc, "req-123")
        
        assert response["error"]["code"] == "VALIDATION_ERROR"
        assert response["error"]["message"] == "Invalid input"
        assert response["error"]["status_code"] == 400
        assert response["error"]["details"] == {"field": "required"}
        assert response["request_id"] == "req-123"
    
    def test_create_error_response_no_request_id(self):
        """Test creating error response without request ID."""
        exc = RateLimitError("Too many requests")
        response = create_error_response(exc)
        
        assert response["error"]["code"] == "RATE_LIMIT_EXCEEDED"
        assert "request_id" not in response


class TestLoggingConfig:
    """Test logging configuration."""
    
    def test_setup_development_logging(self):
        """Test development logging setup."""
        setup_logging("development")
        logger = get_logger("test_logger")
        
        # Should not raise any errors
        logger.info("Test message")
        assert logger.name == "test_logger"
    
    def test_setup_testing_logging(self):
        """Test testing logging setup."""
        setup_logging("testing")
        logger = get_logger("test_logger")
        
        # Should not raise any errors
        logger.warning("Test warning")
        assert logger.name == "test_logger"
    
    def test_log_with_context(self):
        """Test logging with context."""
        setup_logging("development")
        logger = get_logger("test_logger")
        
        # Test logging with context
        log_with_context(
            logger, logging.INFO, "Test message",
            tenant_id="CWFM",
            session_id="user-123",
            request_id="req-456"
        )
        
        # Should not raise any errors
        assert logger.name == "test_logger"


class TestErrorHandlingIntegration:
    """Test error handling integration with FastAPI."""
    
    @pytest.fixture
    def app(self):
        """Create a test FastAPI app with error handlers."""
        from core.logging_config import initialize_logging
        from core.error_handling import (
            rag_exception_handler, validation_exception_handler,
            http_exception_handler, general_exception_handler
        )
        from fastapi.exceptions import RequestValidationError
        from starlette.exceptions import HTTPException as StarletteHTTPException
        
        initialize_logging("testing")
        app = FastAPI()
        
        # Add exception handlers
        app.add_exception_handler(RAGException, rag_exception_handler)
        app.add_exception_handler(RequestValidationError, validation_exception_handler)
        app.add_exception_handler(StarletteHTTPException, http_exception_handler)
        app.add_exception_handler(Exception, general_exception_handler)
        
        @app.get("/test-rag-exception")
        async def test_rag_exception():
            raise ValidationError("Test validation error", {"field": "test"})
        
        @app.get("/test-http-exception")
        async def test_http_exception():
            raise StarletteHTTPException(status_code=404, detail="Not found")
        
        @app.get("/test-general-exception")
        async def test_general_exception():
            raise ValueError("Test general error")
        
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    def test_rag_exception_handler(self, client):
        """Test RAG exception handler."""
        response = client.get("/test-rag-exception")
        
        assert response.status_code == 400
        data = response.json()
        assert data["error"]["code"] == "VALIDATION_ERROR"
        assert data["error"]["message"] == "Test validation error"
        assert data["error"]["status_code"] == 400
        assert data["error"]["details"] == {"field": "test"}
    
    def test_http_exception_handler(self, client):
        """Test HTTP exception handler."""
        response = client.get("/test-http-exception")
        
        assert response.status_code == 404
        data = response.json()
        assert data["error"]["code"] == "HTTP_ERROR"
        assert data["error"]["message"] == "Not found"
        assert data["error"]["status_code"] == 404
    
    def test_general_exception_handler(self, client):
        """Test general exception handler."""
        response = client.get("/test-general-exception")
        
        assert response.status_code == 500
        data = response.json()
        assert data["error"]["code"] == "INTERNAL_ERROR"
        assert data["error"]["message"] == "An unexpected error occurred"
        assert data["error"]["status_code"] == 500


class TestMiddleware:
    """Test middleware functionality."""
    
    @pytest.fixture
    def app(self):
        """Create a test FastAPI app with middleware."""
        from core.logging_config import initialize_logging
        from core.middleware import RequestCorrelationMiddleware, MetricsMiddleware
        
        initialize_logging("testing")
        app = FastAPI()
        
        # Add middleware
        app.add_middleware(RequestCorrelationMiddleware)
        metrics_middleware = MetricsMiddleware(app)
        app.add_middleware(MetricsMiddleware)
        
        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}
        
        return app, metrics_middleware
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        app_instance, _ = app
        return TestClient(app_instance)
    
    def test_request_correlation_headers(self, client):
        """Test that correlation headers are added."""
        response = client.get("/test?tenant_id=CWFM&session_id=user-123")
        
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        assert "X-Tenant-ID" in response.headers
        assert response.headers["X-Tenant-ID"] == "CWFM"
    
    def test_metrics_collection(self, app):
        """Test metrics collection."""
        _, metrics_middleware = app
        client = TestClient(app[0])
        
        # Make some requests
        client.get("/test")
        client.get("/test")
        client.get("/test")
        
        metrics = metrics_middleware.get_metrics()
        
        assert metrics["total_requests"] == 3
        assert metrics["total_errors"] == 0
        assert metrics["error_rate"] == 0.0
        assert metrics["average_response_time"] > 0
