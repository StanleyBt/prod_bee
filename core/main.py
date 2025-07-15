# core/main.py

import logging
import uvicorn
import sys
import warnings
from pathlib import Path

# Suppress OpenTelemetry context warnings
warnings.filterwarnings("ignore", message=".*was created in a different Context.*")
warnings.filterwarnings("ignore", message=".*Failed to detach context.*")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.app import app 
from core.config import (
    DEFAULT_MODULE,
    DELETE_COLLECTION_ON_INGEST,
    WEAVIATE_COLLECTION_NAME,
)
from services.per_tenant_storage import (
    initialize_per_tenant_storage,
    list_tenant_collections,
)

from services.llm import test_llm_connection # To test LLM connection before starting
from core.ingestion import ingest_all_documents # The actual ingestion logic

# Basic logging setup for the main script
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def pre_flight_checks_and_ingestion():
    """
    Performs initial checks for services and triggers ingestion if necessary.
    This function can be run once before starting the API server,
    or as a separate script/command.
    """
    logger.info("Performing pre-flight checks and initial data setup...")

    # Test LLM connection
    if not test_llm_connection():
        logger.error("LLM connection failed. This is critical for the application.")
        # Depending on criticality, you might want to exit here
        # sys.exit(1)

    # Initialize per-tenant storage for schema management and potential ingestion
    # Note: This is a temporary initialization for ingestion only
    # The main service initialization happens in the FastAPI lifespan
    storage_ready = initialize_per_tenant_storage()
    if not storage_ready:
        logger.error("Per-tenant storage not ready. Data ingestion and retrieval will fail.")
        # sys.exit(1) # Consider exiting if storage is absolutely essential

    # Memory storage is handled by per-tenant storage
    if not storage_ready:
        logger.warning("Storage not ready. Conversation memory and document retrieval may fail.")

    # Manage per-tenant collections and trigger ingestion
    if storage_ready:
        # List existing collections for info
        existing_collections = list_tenant_collections()
        if existing_collections:
            logger.info(f"Found existing tenant collections: {existing_collections}")
        else:
            logger.info("No existing tenant collections found. New collections will be created during ingestion.")

        # Trigger data ingestion
        logger.info("Starting data ingestion process (if new/changed files are detected)...")
        # ingest_all_documents expects 'data' as base_path by default.
        # It handles file hashing internally to avoid re-ingesting unchanged files.
        ingest_all_documents()
        logger.info("Initial data ingestion process completed.")
    else:
        logger.warning("Skipping data ingestion as per-tenant storage is not initialized.")

    # Close the temporary storage connection used for ingestion
    from services.per_tenant_storage import close_per_tenant_storage
    close_per_tenant_storage()

    logger.info("Pre-flight checks and initial data setup complete.")

if __name__ == "__main__":
    # Perform pre-flight checks and ingestion before starting the API server
    pre_flight_checks_and_ingestion()

    logger.info("Starting FastAPI application with Uvicorn...")
    # Start the FastAPI application.
    # The 'app' object is imported from 'api.app'.
    uvicorn.run(app, host="0.0.0.0", port=8000)

