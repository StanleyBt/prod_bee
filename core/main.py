# core/main.py

import logging
import uvicorn
import sys
from pathlib import Path


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.app import app 
from core.config import (
    DEFAULT_MODULE,
    DELETE_COLLECTION_ON_INGEST,
    WEAVIATE_COLLECTION_NAME,
)
from services.vector_db import (
    initialize_weaviate,
    create_collection_schema,
    delete_collection,
    collection_exists,
)
from services.llm import test_llm_connection # To test LLM connection before starting
from services.memory_store import initialize_mem0 # To test Mem0 connection before starting
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

    # Initialize Weaviate for schema management and potential ingestion
    weaviate_ready = initialize_weaviate()
    if not weaviate_ready:
        logger.error("Weaviate not ready. Data ingestion and retrieval will fail.")
        # sys.exit(1) # Consider exiting if Weaviate is absolutely essential

    # Initialize Mem0 client (only if it needs a separate initialization from lifespan)
    # Note: lifespan already initializes Mem0, this is more for a pre-check if needed.
    mem0_ready = initialize_mem0()
    if not mem0_ready:
        logger.warning("Mem0 memory store not ready. Conversation history might not be stored/retrieved.")

    # Manage Weaviate collection schema and trigger ingestion
    if weaviate_ready:
        if DELETE_COLLECTION_ON_INGEST:
            logger.info(f"Deleting existing Weaviate collection '{WEAVIATE_COLLECTION_NAME}' as per config...")
            delete_collection()
            logger.info(f"Collection '{WEAVIATE_COLLECTION_NAME}' deleted.")

        if not collection_exists():
            logger.info(f"Collection '{WEAVIATE_COLLECTION_NAME}' does not exist, creating schema...")
            create_collection_schema()
            logger.info(f"Collection '{WEAVIATE_COLLECTION_NAME}' schema created.")
        else:
            logger.info(f"Collection '{WEAVIATE_COLLECTION_NAME}' already exists.")

        # Trigger data ingestion
        logger.info("Starting data ingestion process (if new/changed files are detected)...")
        # ingest_all_documents expects 'data' as base_path by default.
        # It handles file hashing internally to avoid re-ingesting unchanged files.
        ingest_all_documents()
        logger.info("Initial data ingestion process completed.")
    else:
        logger.warning("Skipping data ingestion as Weaviate is not initialized.")

    logger.info("Pre-flight checks and initial data setup complete.")

if __name__ == "__main__":
    # Perform pre-flight checks and ingestion before starting the API server
    pre_flight_checks_and_ingestion()

    logger.info("Starting FastAPI application with Uvicorn...")
    # Start the FastAPI application.
    # The 'app' object is imported from 'api.app'.
    uvicorn.run(app, host="0.0.0.0", port=8000)

