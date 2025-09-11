"""
Input Validation and Sanitization Module

Provides comprehensive input validation and sanitization to prevent injection attacks
and ensure data integrity across the RAG API.
"""

import re
import html
import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, field_validator, Field
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

class SanitizedQueryRequest(BaseModel):
    """Sanitized and validated query request."""
    input: str = Field(..., min_length=1, max_length=1000, description="User query")
    tenant_id: str = Field(..., min_length=1, max_length=50, description="Tenant ID")
    session_id: str = Field(..., min_length=1, max_length=100, description="Session ID")
    module: Optional[str] = Field(None, min_length=1, max_length=50, description="Module name")
    role: str = Field(..., min_length=1, max_length=20, description="User role")

    @field_validator('input')
    @classmethod
    def sanitize_input(cls, v):
        """Sanitize user input to prevent injection attacks."""
        if not v or not v.strip():
            raise ValueError("Input cannot be empty")
        
        # Remove HTML tags
        v = re.sub(r'<[^>]+>', '', v)
        
        # Escape HTML entities
        v = html.escape(v)
        
        # Remove potentially dangerous characters
        v = re.sub(r'[<>"\']', '', v)
        
        # Limit consecutive spaces
        v = re.sub(r'\s+', ' ', v)
        
        return v.strip()

    @field_validator('tenant_id')
    @classmethod
    def validate_tenant_id(cls, v):
        """Validate tenant ID format."""
        if not re.match(r'^[A-Z0-9_-]+$', v):
            raise ValueError("Tenant ID must contain only uppercase letters, numbers, underscores, and hyphens")
        return v.upper()

    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v):
        """Validate session ID format."""
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Session ID must contain only alphanumeric characters, underscores, and hyphens")
        return v

    @field_validator('module')
    @classmethod
    def validate_module(cls, v):
        """Validate module name format."""
        if v is not None:
            if not re.match(r'^[a-zA-Z0-9_-]+$', v):
                raise ValueError("Module name must contain only alphanumeric characters, underscores, and hyphens")
        return v

    @field_validator('role')
    @classmethod
    def validate_role(cls, v):
        """Validate role format."""
        valid_roles = ['admin', 'hr', 'employee', 'contractor', 'viewer']
        if v.lower() not in valid_roles:
            raise ValueError(f"Role must be one of: {', '.join(valid_roles)}")
        return v.lower()

class SanitizedClearRequest(BaseModel):
    """Sanitized and validated clear conversations request."""
    tenant_id: str = Field(..., min_length=1, max_length=50)
    session_id: str = Field(..., min_length=1, max_length=100)

    @field_validator('tenant_id')
    @classmethod
    def validate_tenant_id(cls, v):
        if not re.match(r'^[A-Z0-9_-]+$', v):
            raise ValueError("Invalid tenant ID format")
        return v.upper()

    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Invalid session ID format")
        return v

def sanitize_text(text: str, max_length: int = 1000) -> str:
    """Sanitize text input to prevent injection attacks."""
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Escape HTML entities
    text = html.escape(text)
    
    # Remove potentially dangerous characters
    text = re.sub(r'[<>"\']', '', text)
    
    # Limit consecutive spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    return text.strip()

def validate_file_path(path: str) -> bool:
    """Validate file path to prevent path traversal attacks."""
    # Check for path traversal attempts
    if '..' in path or '//' in path:
        return False
    
    # Check for absolute paths
    if path.startswith('/') or path.startswith('\\'):
        return False
    
    # Check for dangerous characters
    dangerous_chars = ['<', '>', ':', '"', '|', '?', '*']
    if any(char in path for char in dangerous_chars):
        return False
    
    return True

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and injection attacks."""
    # Remove path separators
    filename = re.sub(r'[\\/]', '', filename)
    
    # Remove dangerous characters
    filename = re.sub(r'[<>:"|?*]', '', filename)
    
    # Limit length
    if len(filename) > 255:
        filename = filename[:255]
    
    return filename

# Note: Removed unused validation functions to simplify the code
# The Pydantic models with validators handle all the validation we need

