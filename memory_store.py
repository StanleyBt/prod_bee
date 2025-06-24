# memory_store.py

import logging
from mem0 import Memory
from config import (
    AZURE_API_KEY,
    AZURE_API_BASE,
    AZURE_API_VERSION,
    AZURE_DEPLOYMENT_NAME,
    AZURE_EMBEDDING_API_KEY,
    AZURE_EMBEDDING_API_BASE,
    AZURE_EMBEDDING_API_VERSION,
    AZURE_EMBEDDING_DEPLOYMENT_NAME,
)

logger = logging.getLogger(__name__)

memory_client = None

def initialize_mem0():
    """Initializes Mem0 with Azure OpenAI embeddings and Weaviate as vector store."""
    global memory_client
    try:
        mem0_config = {
            "vector_store": {
                "provider": "weaviate",
                "config": {
                    "cluster_url": "http://localhost:8080",
                    "collection_name": "Mem0Memory",
                    "embedding_model_dims": 1536
                }
            },
            "llm": {
                "provider": "azure_openai",
                "config": {
                    "model": AZURE_DEPLOYMENT_NAME,
                    "temperature": 0.1,
                    "max_tokens": 800,
                    "azure_kwargs": {
                        "azure_deployment": AZURE_DEPLOYMENT_NAME,
                        "api_version": AZURE_API_VERSION,
                        "azure_endpoint": AZURE_API_BASE,
                        "api_key": AZURE_API_KEY
                    }
                }
            },
            "embedder": {
                "provider": "azure_openai",
                "config": {
                    "model": AZURE_EMBEDDING_DEPLOYMENT_NAME,
                    "azure_kwargs": {
                        "azure_deployment": AZURE_EMBEDDING_DEPLOYMENT_NAME,
                        "api_version": AZURE_EMBEDDING_API_VERSION,
                        "azure_endpoint": AZURE_EMBEDDING_API_BASE,
                        "api_key": AZURE_EMBEDDING_API_KEY
                    }
                }
            }
        }
        memory_client = Memory.from_config(mem0_config)
        logger.info("Mem0 initialized with Azure OpenAI embeddings and Weaviate")
        return True
    except Exception as e:
        logger.error(f"Mem0 initialization failed: {e}")
        return False

def store_in_mem0(user_input, response_text, span=None):
    """Store a conversation turn in Mem0."""
    if not memory_client:
        logger.warning("Mem0 not available for storage")
        return
    try:
        messages = [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": response_text}
        ]
        memory_result = memory_client.add(messages, user_id="prod-bee-session")
        logger.info("Stored in Mem0.")
        if span:
            try:
                span.update(metadata={"mem0_status": "Stored", "mem0_result": str(memory_result)})
            except Exception:
                pass
    except Exception as e:
        logger.error(f"Mem0 storage failed: {e}")
        if span:
            try:
                span.update(metadata={"mem0_error": str(e)})
            except Exception:
                pass

def retrieve_mem0_memories():
    """Retrieve all memories for the session."""
    if not memory_client:
        logger.warning("Mem0 not available for retrieval")
        return []
    try:
        memories = memory_client.get_all(user_id="prod-bee-session")
        logger.info(f"Retrieved {len(memories)} memories.")
        return memories
    except Exception as e:
        logger.error(f"Failed to retrieve memories: {e}")
        return []
