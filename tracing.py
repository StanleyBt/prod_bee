# tracing.py

import logging
import os
from langfuse import get_client

logger = logging.getLogger(__name__)

_langfuse_client = None

def get_langfuse_client():
    """Initializes and returns the Langfuse client (singleton)."""
    global _langfuse_client
    
    # Check if tracing is disabled via environment variable
    if os.getenv("DISABLE_TRACING", "false").lower() == "true":
        logger.info("Tracing disabled via DISABLE_TRACING environment variable")
        return None
    
    if _langfuse_client is not None:
        return _langfuse_client
    try:
        # Initialize with basic configuration
        _langfuse_client = get_client()
        logger.info("Langfuse client initialized")
        return _langfuse_client
    except Exception as e:
        logger.warning(f"Langfuse initialization failed: {e}")
        _langfuse_client = None
        return None
