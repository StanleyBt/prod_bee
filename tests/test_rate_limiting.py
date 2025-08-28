"""
Unit tests for rate limiting module.
"""

import pytest
import time
from unittest.mock import patch, MagicMock
from fastapi import HTTPException
from core.error_handling import RateLimitError

from core.rate_limiting import (
    RateLimitConfig, RateLimiter,
    get_rate_limit_key, create_rate_limiter, check_user_rate_limit
)

class TestRateLimitConfig:
    """Test rate limit configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RateLimitConfig()
        
        assert config.requests_per_minute == 60
        assert config.requests_per_hour == 1000
        assert config.requests_per_day == 10000
        assert config.burst_limit == 10
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = RateLimitConfig(
            requests_per_minute=30,
            requests_per_hour=500,
            requests_per_day=5000,
            burst_limit=5
        )
        
        assert config.requests_per_minute == 30
        assert config.requests_per_hour == 500
        assert config.requests_per_day == 5000
        assert config.burst_limit == 5

class TestRateLimiter:
    """Test rate limiter functionality."""
    
    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization."""
        config = RateLimitConfig(requests_per_minute=10)
        limiter = RateLimiter(config)
        
        assert limiter.config.requests_per_minute == 10
        assert len(limiter.windows) == 0
    
    def test_check_rate_limit_within_limits(self):
        """Test rate limit check when within limits."""
        config = RateLimitConfig(requests_per_minute=5)
        limiter = RateLimiter(config)
        
        # Make requests within limit
        for i in range(3):
            within_limits, counts = limiter.check_and_record("test_key")
            assert within_limits == True
            assert counts['minute'] == i + 1
    
    def test_check_rate_limit_exceeded(self):
        """Test rate limit check when exceeded."""
        config = RateLimitConfig(requests_per_minute=2)
        limiter = RateLimiter(config)
        
        # Make requests within limit
        within_limits, counts = limiter.check_and_record("test_key")
        assert within_limits == True
        assert counts['minute'] == 1
        
        within_limits, counts = limiter.check_and_record("test_key")
        assert within_limits == True
        assert counts['minute'] == 2
        
        # Exceed limit
        within_limits, counts = limiter.check_and_record("test_key")
        assert within_limits == False
        assert counts['minute'] == 2  # Should not increment
    
    def test_rate_limit_window_cleanup(self):
        """Test that old entries are cleaned up from windows."""
        config = RateLimitConfig(requests_per_minute=5)
        limiter = RateLimiter(config)
        
        # Add old entries
        old_time = time.time() - 70  # 70 seconds ago
        limiter.windows["test_key"]["minute"].append(old_time)
        
        # Check rate limit (should clean up old entries)
        within_limits, counts = limiter.check_and_record("test_key")
        assert within_limits == True
        assert counts['minute'] == 1  # Only the new request
    
    def test_multiple_keys(self):
        """Test that different keys have separate rate limits."""
        config = RateLimitConfig(requests_per_minute=2)
        limiter = RateLimiter(config)
        
        # Use up limit for key1
        limiter.check_and_record("key1")
        limiter.check_and_record("key1")
        within_limits, counts = limiter.check_and_record("key1")
        assert within_limits == False
        
        # Key2 should still be within limits
        within_limits, counts = limiter.check_and_record("key2")
        assert within_limits == True
        assert counts['minute'] == 1

# Note: Removed middleware tests as we simplified to endpoint-level rate limiting

class TestRateLimitUtilities:
    """Test rate limiting utility functions."""
    
    def test_get_rate_limit_key(self):
        """Test rate limit key generation."""
        key = get_rate_limit_key("CWFM", "user123", "/query")
        assert key == "CWFM:user123:/query"
        
        key = get_rate_limit_key("TEST", "admin", "/welcome")
        assert key == "TEST:admin:/welcome"
    
    def test_create_rate_limiter(self):
        """Test rate limiter creation."""
        limiter = create_rate_limiter()
        
        assert isinstance(limiter, RateLimiter)
        assert limiter.config.requests_per_minute == 60
        assert limiter.config.requests_per_hour == 1000
        assert limiter.config.requests_per_day == 10000
    
    def test_check_user_rate_limit_success(self):
        """Test user rate limit check when within limits."""
        # This would use the global rate_limiter
        # For testing, we'll mock it
        with patch('core.rate_limiting.rate_limiter') as mock_limiter:
            mock_limiter.check_and_record.return_value = (True, {'minute': 1, 'hour': 5, 'day': 50})
            
            result = check_user_rate_limit("CWFM", "user123", "/query")
            assert result == True
            
            # Verify the key was generated correctly
            mock_limiter.check_and_record.assert_called_with("CWFM:user123:/query")
    
    def test_check_user_rate_limit_exceeded(self):
        """Test user rate limit check when exceeded."""
        with patch('core.rate_limiting.rate_limiter') as mock_limiter:
            mock_limiter.check_and_record.return_value = (False, {'minute': 60, 'hour': 1000, 'day': 10000})
            
            with pytest.raises(RateLimitError) as exc_info:
                check_user_rate_limit("CWFM", "user123", "/query")
            
            assert exc_info.value.status_code == 429
            assert "Rate limit exceeded" in str(exc_info.value.message)

class TestRateLimitIntegration:
    """Integration tests for rate limiting."""
    
    def test_full_rate_limit_flow(self):
        """Test complete rate limiting flow."""
        config = RateLimitConfig(requests_per_minute=3)
        limiter = RateLimiter(config)
        
        key = "test_user:test_endpoint"
        
        # First three requests should succeed
        for i in range(3):
            within_limits, counts = limiter.check_and_record(key)
            assert within_limits == True
            assert counts['minute'] == i + 1
        
        # Fourth request should fail
        within_limits, counts = limiter.check_and_record(key)
        assert within_limits == False
        assert counts['minute'] == 3  # Should not increment
    
    def test_rate_limit_recovery(self):
        """Test that rate limits recover after time passes."""
        config = RateLimitConfig(requests_per_minute=2)
        limiter = RateLimiter(config)
        
        key = "test_user:test_endpoint"
        
        # Use up the limit
        limiter.check_and_record(key)
        limiter.check_and_record(key)
        within_limits, counts = limiter.check_and_record(key)
        assert within_limits == False
        
        # Simulate time passing by manually clearing the window
        limiter.windows[key]['minute'].clear()
        
        # Should be able to make requests again
        within_limits, counts = limiter.check_and_record(key)
        assert within_limits == True
        assert counts['minute'] == 1

