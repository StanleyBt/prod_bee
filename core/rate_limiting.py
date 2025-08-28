"""
Rate Limiting Module

Provides rate limiting functionality to prevent API abuse and ensure fair usage.
Uses sliding window algorithm with in-memory storage (replace with Redis in production).
"""

import time
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
from fastapi import HTTPException, status
from core.error_handling import RateLimitError
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10  # Allow burst of requests

class RateLimiter:
    """Rate limiter using sliding window algorithm."""
    
    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        self.windows: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: {
                'minute': deque(),
                'hour': deque(),
                'day': deque()
            }
        )
    
    def _clean_old_entries(self, key: str, window_type: str, current_time: float, window_size: int):
        """Remove old entries from the window."""
        window = self.windows[key][window_type]
        while window and (current_time - window[0]) > window_size:
            window.popleft()
    
    def _check_rate_limit(self, key: str, current_time: float) -> Tuple[bool, Dict[str, int]]:
        """Check if request is within rate limits."""
        windows = self.windows[key]
        
        # Clean old entries
        self._clean_old_entries(key, 'minute', current_time, 60)
        self._clean_old_entries(key, 'hour', current_time, 3600)
        self._clean_old_entries(key, 'day', current_time, 86400)
        
        # Count requests in each window
        minute_count = len(windows['minute'])
        hour_count = len(windows['hour'])
        day_count = len(windows['day'])
        
        # Check limits
        within_limits = (
            minute_count < self.config.requests_per_minute and
            hour_count < self.config.requests_per_hour and
            day_count < self.config.requests_per_day
        )
        
        return within_limits, {
            'minute': minute_count,
            'hour': hour_count,
            'day': day_count
        }
    
    def _add_request(self, key: str, current_time: float):
        """Add a request to the rate limiting windows."""
        windows = self.windows[key]
        windows['minute'].append(current_time)
        windows['hour'].append(current_time)
        windows['day'].append(current_time)
    
    def check_and_record(self, key: str) -> Tuple[bool, Dict[str, int]]:
        """Check rate limit and record the request if within limits."""
        current_time = time.time()
        
        # Check if within limits
        within_limits, counts = self._check_rate_limit(key, current_time)
        
        if within_limits:
            # Record the request
            self._add_request(key, current_time)
            # Get updated counts after recording
            updated_counts = {
                'minute': len(self.windows[key]['minute']),
                'hour': len(self.windows[key]['hour']),
                'day': len(self.windows[key]['day'])
            }
            logger.debug(f"Rate limit check passed for {key}: {updated_counts}")
            return within_limits, updated_counts
        else:
            logger.warning(f"Rate limit exceeded for {key}: {counts}")
            return within_limits, counts

# Note: Removed RateLimitMiddleware as it was overcomplicated
# We use endpoint-level rate limiting instead, which is simpler and more reliable

def get_rate_limit_key(tenant_id: str, user_id: str, endpoint: str) -> str:
    """Generate a rate limit key based on tenant, user, and endpoint."""
    return f"{tenant_id}:{user_id}:{endpoint}"

def create_rate_limiter() -> RateLimiter:
    """Create a rate limiter with default configuration."""
    config = RateLimitConfig(
        requests_per_minute=60,
        requests_per_hour=1000,
        requests_per_day=10000,
        burst_limit=10
    )
    return RateLimiter(config)

# Global rate limiter instance
rate_limiter = create_rate_limiter()

def check_user_rate_limit(tenant_id: str, user_id: str, endpoint: str) -> bool:
    """Check rate limit for a specific user and endpoint."""
    key = get_rate_limit_key(tenant_id, user_id, endpoint)
    within_limits, counts = rate_limiter.check_and_record(key)
    
    if not within_limits:
        logger.warning(f"User rate limit exceeded: {key}")
        raise RateLimitError(f"Rate limit exceeded. Current usage: {counts}")
    
    return True

