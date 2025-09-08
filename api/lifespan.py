# api/lifespan.py

import logging
import signal
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI

from services.per_tenant_storage import initialize_per_tenant_storage, close_per_tenant_storage
from services.mem0_service import initialize_mem0_service, close_mem0_service
from utils.embeddings import shutdown_embedding_cache

logger = logging.getLogger(__name__)

def signal_handler(signum, frame):
    """Handle shutdown signals to ensure proper cleanup"""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    close_per_tenant_storage()
    close_mem0_service()
    shutdown_embedding_cache()
    sys.exit(0)

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)



@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI application.
    Initializes per-tenant storage which handles both document chunks and memory storage.
    Note: This will create fresh connections even if they were closed by main.py pre-flight checks.
    """

    
    logger.info("Application lifespan startup: Initializing services...")
    
    # Initialize per-tenant storage (handles both document chunks and memory storage)
    try:
        storage_ready = initialize_per_tenant_storage()
        if storage_ready:
            logger.info("Per-tenant storage initialized successfully (includes memory storage).")
        else:
            logger.error("Failed to initialize per-tenant storage.")
    except Exception as e:
        logger.error(f"Error during per-tenant storage initialization: {e}")
        storage_ready = False

    # Initialize Mem0 service for user preference storage
    try:
        mem0_ready = initialize_mem0_service()
        if mem0_ready:
            logger.info("Mem0 service initialized successfully for user preference management.")
        else:
            logger.warning("Mem0 service not ready. User preference storage and retrieval will not work.")
    except Exception as e:
        logger.error(f"Error during Mem0 service initialization: {e}")
        mem0_ready = False



    yield # Application runs here

    logger.info("Application lifespan shutdown: Closing services...")
    

    
    # Close per-tenant storage
    try:
        close_per_tenant_storage()
        logger.info("Per-tenant storage connection closed.")
    except Exception as e:
        logger.error(f"Error during per-tenant storage shutdown: {e}")

    # Close Mem0 service
    try:
        close_mem0_service()
        logger.info("Mem0 service connections closed")
    except Exception as e:
        logger.error(f"Error during Mem0 service shutdown: {e}")

    # Close embedding cache
    try:
        shutdown_embedding_cache()
        logger.info("Embedding cache shutdown complete")
    except Exception as e:
        logger.error(f"Error during embedding cache shutdown: {e}")

    logger.info("Application shutdown complete.")

