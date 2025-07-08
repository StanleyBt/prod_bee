import logging
from typing import Optional, List, Dict, Any
from mem0 import Memory
from core.config import (
    AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION, AZURE_DEPLOYMENT_NAME,
    AZURE_EMBEDDING_API_KEY, AZURE_EMBEDDING_API_BASE, AZURE_EMBEDDING_API_VERSION, AZURE_EMBEDDING_DEPLOYMENT_NAME,
    WEAVIATE_URL,
)
import json

logger = logging.getLogger(__name__)

memory_client: Optional[Memory] = None

# Default values for session_id and module if not provided
DEFAULT_SESSION_ID = "default_session"
DEFAULT_MODULE = "general"

def initialize_mem0() -> bool:
    """
    Initializes Mem0 with Azure OpenAI embeddings and Weaviate as vector store.
    Uses configuration from core.config.
    """
    global memory_client
    try:
        mem0_config = {
            "vector_store": {
                "provider": "weaviate",
                "config": {
                    "cluster_url": WEAVIATE_URL,
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

def _get_memory_key(tenant_id: str, session_id: Optional[str] = None, module: Optional[str] = None) -> str:
    """
    Generates a unique memory key based on tenant_id, session_id, and module.
    Ensures multi-tenancy and module-level isolation.
    """
    session_id = session_id if session_id is not None else DEFAULT_SESSION_ID
    module = module if module is not None else DEFAULT_MODULE
    return f"{tenant_id}:{module}:{session_id}"

def store_in_mem0(
    tenant_id: str,
    user_input: str,
    response_text: str,
    session_id: Optional[str] = None,
    module: Optional[str] = None,
    span: Optional[object] = None
) -> None:
    """
    Store a conversation turn in Mem0 using a tenant/module/session specific key.
    """
    if not memory_client:
        logger.warning("Mem0 not available for storage")
        return

    mem_key = _get_memory_key(tenant_id, session_id, module)
    try:
        messages = [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": response_text},
        ]
        memory_result = memory_client.add(messages, user_id=mem_key)
        logger.info(f"Stored conversation in Mem0 for key: {mem_key}")
        
        if span:
            try:
                span.update(metadata={"mem0_status": "Stored", "mem0_result": str(memory_result)})
            except Exception:
                pass
    except Exception as e:
        logger.error(f"Mem0 storage failed for key '{mem_key}': {e}")
        if span:
            try:
                span.update(metadata={"mem0_error": str(e)})
            except Exception:
                pass

def retrieve_mem0_memories(
    tenant_id: str,
    session_id: Optional[str] = None,
    module: Optional[str] = None
) -> List[Dict]:
    """
    Retrieve all memories for a specific tenant, session, and module.
    """
    if not memory_client:
        logger.warning("Mem0 not available for retrieval")
        return []

    mem_key = _get_memory_key(tenant_id, session_id, module)
    try:
        result = memory_client.get_all(user_id=mem_key)
        memories = result.get("results", [])
        logger.info(f"Retrieved {len(memories)} memories for key: {mem_key}")
        return memories
    except Exception as e:
        logger.error(f"Failed to retrieve memories for key '{mem_key}': {e}")
        return []

