# api/lifespan.py

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from services.per_tenant_storage import initialize_per_tenant_storage, close_per_tenant_storage

logger = logging.getLogger(__name__)

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
            logger.info("✓ Per-tenant storage initialized successfully (includes memory storage).")
        else:
            logger.error("❌ Failed to initialize per-tenant storage.")
    except Exception as e:
        logger.error(f"❌ Error during per-tenant storage initialization: {e}")
        storage_ready = False

    yield # Application runs here

    logger.info("Application lifespan shutdown: Closing services...")
    
    # Close per-tenant storage
    try:
        close_per_tenant_storage()
        logger.info("✓ Per-tenant storage connection closed.")
    except Exception as e:
        logger.error(f"❌ Error during per-tenant storage shutdown: {e}")

    # Memory cleanup is handled by per-tenant storage

    logger.info("Application shutdown complete.")

