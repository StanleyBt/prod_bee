"""
Resource Management Utilities

Provides context managers and utilities for proper resource management
across the RAG API services.
"""

import logging
import threading
from contextlib import contextmanager
from typing import Optional, Generator
from functools import wraps

logger = logging.getLogger(__name__)

class ResourceManager:
    """
    Centralized resource manager for handling service connections.
    """
    
    def __init__(self):
        self._locks = {}
        self._initialized_services = set()
    
    def get_lock(self, service_name: str) -> threading.Lock:
        """Get or create a lock for a specific service."""
        if service_name not in self._locks:
            self._locks[service_name] = threading.Lock()
        return self._locks[service_name]
    
    def mark_service_initialized(self, service_name: str):
        """Mark a service as initialized."""
        self._initialized_services.add(service_name)
        logger.debug(f"Service {service_name} marked as initialized")
    
    def mark_service_uninitialized(self, service_name: str):
        """Mark a service as uninitialized."""
        self._initialized_services.discard(service_name)
        logger.debug(f"Service {service_name} marked as uninitialized")
    
    def is_service_initialized(self, service_name: str) -> bool:
        """Check if a service is initialized."""
        return service_name in self._initialized_services
    
    def get_initialized_services(self) -> set:
        """Get all initialized services."""
        return self._initialized_services.copy()

# Global resource manager instance
resource_manager = ResourceManager()

@contextmanager
def service_lock(service_name: str) -> Generator[threading.Lock, None, None]:
    """
    Context manager for service-specific locking.
    
    Args:
        service_name: Name of the service to lock
        
    Yields:
        The lock for the service
    """
    lock = resource_manager.get_lock(service_name)
    logger.debug(f"Acquiring lock for {service_name}")
    try:
        with lock:
            logger.debug(f"Lock acquired for {service_name}")
            yield lock
    finally:
        logger.debug(f"Lock released for {service_name}")

def with_service_lock(service_name: str):
    """
    Decorator to automatically acquire service lock for function execution.
    
    Args:
        service_name: Name of the service to lock
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with service_lock(service_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

@contextmanager
def safe_service_operation(service_name: str, operation_name: str = "operation") -> Generator[None, None, None]:
    """
    Context manager for safe service operations with proper error handling.
    
    Args:
        service_name: Name of the service
        operation_name: Name of the operation being performed
    """
    logger.debug(f"Starting {operation_name} for {service_name}")
    try:
        with service_lock(service_name):
            yield
        logger.debug(f"Completed {operation_name} for {service_name}")
    except Exception as e:
        logger.error(f"Error during {operation_name} for {service_name}: {e}")
        raise

def get_service_health_status() -> dict:
    """
    Get health status of all services.
    
    Returns:
        Dictionary with service health status
    """
    return {
        "initialized_services": list(resource_manager.get_initialized_services()),
        "total_services": len(resource_manager.get_initialized_services()),
        "locks": list(resource_manager._locks.keys())
    }

def cleanup_all_resources():
    """
    Cleanup all managed resources.
    """
    logger.info("Cleaning up all managed resources...")
    
    # Clear all initialized services
    resource_manager._initialized_services.clear()
    
    # Clear all locks
    resource_manager._locks.clear()
    
    logger.info("All managed resources cleaned up")

# Health check utilities
def check_service_health(service_name: str, health_check_func: callable) -> bool:
    """
    Check health of a specific service.
    
    Args:
        service_name: Name of the service
        health_check_func: Function to call for health check
        
    Returns:
        True if service is healthy, False otherwise
    """
    try:
        with service_lock(service_name):
            return health_check_func()
    except Exception as e:
        logger.warning(f"Health check failed for {service_name}: {e}")
        return False
