"""
Unit tests for input validation and sanitization module.
"""

import pytest
from fastapi import HTTPException

from core.validation import (
    SanitizedQueryRequest, SanitizedClearRequest, sanitize_text,
    validate_file_path, sanitize_filename
)

class TestSanitizedQueryRequest:
    """Test sanitized query request validation."""
    
    def test_valid_request(self):
        """Test valid query request."""
        request = SanitizedQueryRequest(
            input="How do I request time off?",
            tenant_id="CWFM",
            session_id="user-123",
            module="Attendance",
            role="employee"
        )
        
        assert request.input == "How do I request time off?"
        assert request.tenant_id == "CWFM"
        assert request.session_id == "user-123"
        assert request.module == "Attendance"
        assert request.role == "employee"
    
    def test_input_sanitization(self):
        """Test input sanitization removes dangerous content."""
        request = SanitizedQueryRequest(
            input="<script>alert('xss')</script>How do I request time off?",
            tenant_id="CWFM",
            session_id="user-123",
            role="employee"
        )
        
        # Should remove HTML tags and escape entities
        assert "<script>" not in request.input
        assert "alert('xss')" not in request.input
    
    def test_tenant_id_validation(self):
        """Test tenant ID validation."""
        # Valid tenant ID
        request = SanitizedQueryRequest(
            input="Test query",
            tenant_id="CWFM_123",
            session_id="user-123",
            role="employee"
        )
        assert request.tenant_id == "CWFM_123"
        
        # Invalid tenant ID
        with pytest.raises(ValueError, match="Tenant ID must contain only"):
            SanitizedQueryRequest(
                input="Test query",
                tenant_id="cwfm@123",  # Invalid characters
                session_id="user-123",
                role="employee"
            )
    
    def test_session_id_validation(self):
        """Test session ID validation."""
        # Valid session ID
        request = SanitizedQueryRequest(
            input="Test query",
            tenant_id="CWFM",
            session_id="user_123-456",
            role="employee"
        )
        assert request.session_id == "user_123-456"
        
        # Invalid session ID
        with pytest.raises(ValueError, match="Session ID must contain only"):
            SanitizedQueryRequest(
                input="Test query",
                tenant_id="CWFM",
                session_id="user@123",  # Invalid characters
                role="employee"
            )
    
    def test_role_validation(self):
        """Test role validation."""
        valid_roles = ["admin", "hr", "employee", "contractor", "viewer"]
        
        for role in valid_roles:
            request = SanitizedQueryRequest(
                input="Test query",
                tenant_id="CWFM",
                session_id="user-123",
                role=role
            )
            assert request.role == role.lower()
        
        # Invalid role
        with pytest.raises(ValueError, match="Role must be one of"):
            SanitizedQueryRequest(
                input="Test query",
                tenant_id="CWFM",
                session_id="user-123",
                role="invalid_role"
            )
    
    def test_input_length_validation(self):
        """Test input length validation."""
        # Valid length
        short_input = "Short query"
        request = SanitizedQueryRequest(
            input=short_input,
            tenant_id="CWFM",
            session_id="user-123",
            role="employee"
        )
        assert request.input == short_input
        
        # Too long input
        long_input = "x" * 2001
        with pytest.raises(ValueError, match="String should have at most"):
            SanitizedQueryRequest(
                input=long_input,
                tenant_id="CWFM",
                session_id="user-123",
                role="employee"
            )

class TestSanitizedClearRequest:
    """Test sanitized clear request validation."""
    
    def test_valid_clear_request(self):
        """Test valid clear request."""
        request = SanitizedClearRequest(
            tenant_id="CWFM",
            session_id="user-123"
        )
        
        assert request.tenant_id == "CWFM"
        assert request.session_id == "user-123"
    
    def test_clear_request_validation(self):
        """Test clear request validation."""
        # Invalid tenant ID
        with pytest.raises(ValueError, match="Invalid tenant ID format"):
            SanitizedClearRequest(
                tenant_id="cwfm@123",
                session_id="user-123"
            )
        
        # Invalid session ID
        with pytest.raises(ValueError, match="Invalid session ID format"):
            SanitizedClearRequest(
                tenant_id="CWFM",
                session_id="user@123"
            )

class TestSanitizationFunctions:
    """Test sanitization utility functions."""
    
    def test_sanitize_text(self):
        """Test text sanitization."""
        # Test HTML removal
        html_text = "<script>alert('xss')</script>Hello world"
        sanitized = sanitize_text(html_text)
        assert "<script>" not in sanitized
        assert "alert('xss')" not in sanitized
        
        # Test HTML entity escaping
        special_chars = "Hello & <world>"
        sanitized = sanitize_text(special_chars)
        assert "&amp;" in sanitized
        # Note: < and > get escaped to &lt; and &gt; but then removed by dangerous char filter
        assert "&lt;" not in sanitized  # Removed by dangerous char filter
        assert "&gt;" not in sanitized  # Removed by dangerous char filter
        
        # Test consecutive spaces
        spaced_text = "Hello    world"
        sanitized = sanitize_text(spaced_text)
        assert "    " not in sanitized
        assert "Hello world" in sanitized
        
        # Test length truncation
        long_text = "x" * 3000
        sanitized = sanitize_text(long_text, max_length=100)
        assert len(sanitized) <= 103  # 100 + "..."
        assert sanitized.endswith("...")
    
    def test_validate_file_path(self):
        """Test file path validation."""
        # Valid paths
        assert validate_file_path("data/CWFM/Attendance/file.pdf") == True
        assert validate_file_path("relative/path/file.txt") == True
        
        # Invalid paths - path traversal
        assert validate_file_path("../../../etc/passwd") == False
        assert validate_file_path("data//file.pdf") == False
        
        # Invalid paths - absolute paths
        assert validate_file_path("/absolute/path") == False
        assert validate_file_path("\\windows\\path") == False
        
        # Invalid paths - dangerous characters
        assert validate_file_path("file<name.txt") == False
        assert validate_file_path("file:name.txt") == False
        assert validate_file_path("file\"name.txt") == False
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        # Test path separator removal
        filename = "file/name.txt"
        sanitized = sanitize_filename(filename)
        assert "/" not in sanitized
        
        # Test dangerous character removal
        filename = "file<name:txt"
        sanitized = sanitize_filename(filename)
        assert "<" not in sanitized
        assert ":" not in sanitized
        
        # Test length limitation
        long_filename = "x" * 300
        sanitized = sanitize_filename(long_filename)
        assert len(sanitized) <= 255
    
# Note: Removed tests for unused validation functions
# The remaining tests cover the core validation functionality

