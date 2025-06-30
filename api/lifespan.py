# api/lifespan.py

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from services.vector_db import initialize_weaviate, close_weaviate
from services.memory_store import initialize_mem0

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI application.
    Initializes Weaviate and Mem0 services.
    """
    logger.info("Application lifespan startup: Initializing services...")
    weaviate_ready = False
    mem0_ready = False

    try:
        weaviate_ready = initialize_weaviate()
        if not weaviate_ready:
            logger.error("Failed to initialize Weaviate vector DB.")
    except Exception as e:
        logger.error(f"Error during Weaviate initialization: {e}")

    try:
        mem0_ready = initialize_mem0()
        if not mem0_ready:
            logger.error("Failed to initialize Mem0 memory store.")
    except Exception as e:
        logger.error(f"Error during Mem0 initialization: {e}")

    if weaviate_ready and mem0_ready:
        logger.info("All services initialized successfully.")
    else:
        logger.warning("Some services failed to initialize. Application might not function correctly.")

    yield # Application runs here

    logger.info("Application lifespan shutdown: Closing services...")
    try:
        close_weaviate()
        logger.info("Weaviate connection closed.")
    except Exception as e:
        logger.error(f"Error during Weaviate shutdown: {e}")

    # No explicit close for Mem0Client, as it manages its own connections.
    logger.info("Application shutdown complete.")

