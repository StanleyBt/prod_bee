# tracing.py

import logging
from langfuse import get_client

logger = logging.getLogger(__name__)

_langfuse_client = None

def get_langfuse_client():
    """Initializes and returns the Langfuse client (singleton)."""
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client
    try:
        _langfuse_client = get_client()
        logger.info("Langfuse client initialized")
        return _langfuse_client
    except Exception as e:
        logger.warning(f"Langfuse initialization failed: {e}")
        _langfuse_client = None
        return None
