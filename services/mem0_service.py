"""
Mem0 Service for Multi-Tenant Memory Management

This service integrates Mem0 with the existing tenant, module, and user structure
to provide intelligent conversation memory storage and retrieval.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from mem0 import Memory
from core.config import (
    AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION,
    AZURE_DEPLOYMENT_NAME, AZURE_EMBEDDING_DEPLOYMENT_NAME,
    MEM0_CLUSTER_URL, MEM0_COLLECTION_NAME,
    AZURE_EMBEDDING_API_KEY, AZURE_EMBEDDING_API_VERSION,
    AZURE_EMBEDDING_API_BASE,
    DEFAULT_MODULE, DEFAULT_SESSION_ID
)

import os
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Global Mem0 client
mem0_client: Optional[Memory] = None

def initialize_mem0_service() -> bool:
    """
    Initialize the Mem0 service with Azure OpenAI configuration.
    """
    global mem0_client
    try:
        # Validate required environment variables
        if not all([AZURE_API_KEY, AZURE_API_BASE, AZURE_DEPLOYMENT_NAME, AZURE_EMBEDDING_DEPLOYMENT_NAME]):
            logger.error("❌ Missing required Azure OpenAI environment variables for Mem0")
            return False
        
        # Mem0 configuration with Azure OpenAI and Weaviate backend
        config = {
            "vector_store": {
                "provider": "weaviate",
                "config": {
                    "collection_name": MEM0_COLLECTION_NAME,
                    "cluster_url": os.getenv("WEAVIATE_URL", "http://localhost:8080"),
                    "auth_client_secret": None,
                }
            },
            "llm": {
                "provider": "azure_openai",
                "config": {
                    "model": AZURE_DEPLOYMENT_NAME,
                    "temperature": 0.1,
                    "max_tokens": 100,
                    "azure_kwargs": {
                        "azure_deployment": AZURE_DEPLOYMENT_NAME,
                        "api_version": AZURE_API_VERSION,
                        "azure_endpoint": AZURE_API_BASE,
                        "api_key": os.environ["AZURE_OPENAI_API_KEY"]
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
                        "api_key": os.environ["AZURE_EMBEDDING_API_KEY"],
                    }
                }
            }
        }
        # Only log non-sensitive config
        logger.info(f"Mem0 config used for initialization (non-sensitive fields): vector_store.provider={config['vector_store']['provider']}, vector_store.collection_name={config['vector_store']['config']['collection_name']}, vector_store.cluster_url={config['vector_store']['config']['cluster_url']}")
        mem0_client = Memory.from_config(config)
        logger.info(f"Mem0 client vector_store type: {type(mem0_client.vector_store)}")
        logger.info(f"Mem0 client vector_store attributes: {dir(mem0_client.vector_store)}")
        logger.info(f"✅ Mem0 service initialized successfully (backend: {getattr(mem0_client.vector_store, 'provider', 'unknown')})")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize Mem0 service: {e}")
        return False

def _get_user_id(tenant_id: str, user_id: str, session_id: Optional[str] = None, module: Optional[str] = None) -> str:
    """
    Generate a unique user ID for Mem0 based on tenant, user, session, and module.
    Format: tenant_id:user_id:module:session_id
    """
    session = session_id or DEFAULT_SESSION_ID
    mod = module or DEFAULT_MODULE
    return f"{tenant_id}:{user_id}:{mod}:{session}"

def store_conversation_memory(
    tenant_id: str,
    user_id: str,
    user_input: str,
    assistant_response: str,
    session_id: Optional[str] = None,
    module: Optional[str] = None,
) -> bool:
    """
    Store a conversation memory using Mem0 with tenant/user/module/session isolation.
    """
    if not mem0_client:
        logger.error("❌ Mem0 client not initialized")
        return False
    
    try:
        # Construct unique key for this memory
        key = f"{tenant_id}:{user_id}:{module or DEFAULT_MODULE}:{session_id or DEFAULT_SESSION_ID}"
        logger.info(f"[Mem0 store] Key: {key} | tenant_id={tenant_id} user_id={user_id} session_id={session_id} module={module}")
        logger.info(f"[Mem0 STORE] user_input: {user_input}")
        logger.info(f"[Mem0 STORE] assistant_response: {assistant_response}")

        # DEBUG: Minimal test with hardcoded messages
        messages = [
            {"role": "user", "content": "test user"},
            {"role": "assistant", "content": "test assistant"}
        ]
        logger.info(f"[Mem0 store] Attempting to store memory with messages: {messages}")
        result = mem0_client.add(
            user_id=key,
            messages=messages
        )
        logger.info(f"[Mem0 store] Mem0 add() result: {result}")
        # Check if the memory was actually stored
        if not result or (isinstance(result, dict) and result.get('results') == []):
            logger.error(f"❌ Mem0 add() returned empty result - memory was NOT stored for {key}")
            return False
        logger.info(f"✅ Stored conversation memory for {key}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to store conversation memory for {tenant_id}:{user_id}:{module}: {e}")
        logger.error(f"❌ Exception type: {type(e).__name__}")
        logger.error(f"❌ Exception details: {str(e)}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return False

def retrieve_conversation_memories(
    tenant_id: str,
    user_id: str,
    session_id: Optional[str] = None,
    module: Optional[str] = None,
    max_memories: int = 5
) -> List[Dict[str, Any]]:
    """
    Retrieve conversation memories for a specific user in a tenant.
    """
    if not mem0_client:
        logger.error("Mem0 client not initialized")
        return []
    try:
        mem0_user_id = _get_user_id(tenant_id, user_id, session_id, module)
        logger.info(f"[Mem0 retrieve] Key: {mem0_user_id} | tenant_id={tenant_id} user_id={user_id} session_id={session_id} module={module}")
        result = mem0_client.get_all(user_id=mem0_user_id)
        if not result or not result.get('results'):
            logger.info(f"No memories found for {mem0_user_id}")
            return []
        memories = []
        for memory in result.get('results', [])[:max_memories]:
            memories.append({
                'memory': memory.get('memory', ''),
                'metadata': memory.get('metadata', {}),
                'created_at': memory.get('created_at', ''),
                'id': memory.get('id', ''),
                'tenant_id': tenant_id,
                'user_id': user_id,
                'module': module or DEFAULT_MODULE,
                'session_id': session_id or DEFAULT_SESSION_ID
            })
        logger.info(f"✅ Retrieved {len(memories)} memories for {mem0_user_id}")
        return memories
    except Exception as e:
        logger.error(f"❌ Failed to retrieve conversation memories: {e}")
        return []

def get_memory_summary(
    tenant_id: str,
    user_id: str,
    session_id: Optional[str] = None,
    module: Optional[str] = None
) -> str:
    """
    Get a summary of user's conversation history.
    """
    memories = retrieve_conversation_memories(tenant_id, user_id, session_id, module)
    if not memories:
        return "No previous conversation history found."
    summary_parts = []
    for memory in memories:
        summary_parts.append(memory.get('memory', ''))
    return " | ".join(summary_parts)

def clear_user_memories(
    tenant_id: str,
    user_id: str,
    session_id: Optional[str] = None,
    module: Optional[str] = None
) -> bool:
    """
    Clear all memories for a specific user.
    """
    if not mem0_client:
        logger.error("Mem0 client not initialized")
        return False
    try:
        mem0_user_id = _get_user_id(tenant_id, user_id, session_id, module)
        mem0_client.delete_all(user_id=mem0_user_id)
        logger.info(f"✅ Cleared all memories for {mem0_user_id}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to clear memories: {e}")
        return False

def get_tenant_memory_stats(tenant_id: str) -> Dict[str, Any]:
    """
    Get memory statistics for a tenant.
    """
    # Implementation depends on Mem0 API
    return {}

def close_mem0_service() -> None:
    """
    Close the Mem0 service if needed.
    """
    global mem0_client
    mem0_client = None
    logger.info("Mem0 service closed") 

if __name__ == "__main__":
    import weaviate
    import os
    from mem0 import Memory
    from dotenv import load_dotenv
    load_dotenv()

    # Connect to local Weaviate instance
    client = weaviate.connect_to_local()

    # Create a test collection if it doesn't exist
    collection_name = "Mem0Memory"
    if not client.collections.exists(collection_name):
        client.collections.create(
            name=collection_name,
            properties=[
                weaviate.classes.config.Property(name="memory", data_type=weaviate.classes.config.DataType.TEXT),
                weaviate.classes.config.Property(name="metadata", data_type=weaviate.classes.config.DataType.TEXT),
                weaviate.classes.config.Property(name="user_id", data_type=weaviate.classes.config.DataType.TEXT),
                weaviate.classes.config.Property(name="created_at", data_type=weaviate.classes.config.DataType.DATE),
            ],
            vectorizer_config=None
        )
        print(f"Created collection: {collection_name}")
    else:
        print(f"Collection already exists: {collection_name}")
    client.close()

    # Use the working config from test_mem0.py
    config = {
    "vector_store": {
        "provider": "weaviate",
        "config": {
            "cluster_url": os.getenv("WEAVIATE_URL", "http://localhost:8080"),
            "collection_name": collection_name,
            "auth_client_secret": None,
        }
    },
    "llm": {
        "provider": "azure_openai",
        "config": {
            "model": "gpt-4.1-mini",
            "temperature": 0.1,
            "max_tokens": 2000,
            "azure_kwargs": {
                  "azure_deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                  "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
                  "azure_endpoint": os.getenv("AZURE_OPENAI_API_BASE"),
                  "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            }
        }
    },
    "embedder": {
        "provider": "azure_openai",
        "config": {
            "model": "text-embedding-ada-002",
            "azure_kwargs": {
                "api_version": os.getenv("AZURE_EMBEDDING_API_VERSION"),
                "azure_deployment": os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME"),
                "azure_endpoint": os.getenv("AZURE_EMBEDDING_API_BASE"),
                "api_key": os.getenv("AZURE_EMBEDDING_API_KEY"),
            }
        }
    }
}

    # Initialize Mem0 service

    mem0_client = Memory.from_config(config)
    test_user_id = "mem0service_test"
    messages = [
        {"role": "user", "content": "Testing Mem0Service add."},
        {"role": "assistant", "content": "This is a response from Mem0Service."}
    ]
    add_result = mem0_client.add(messages, user_id=test_user_id, metadata={"category": "service_test"})
    print("Add result:", add_result)
    result = mem0_client.get_all(user_id=test_user_id)
    print("Get all result:", result) 