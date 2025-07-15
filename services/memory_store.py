"""
Memory Store Service using Weaviate

This module provides conversation memory storage using Weaviate as the vector store.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from collections import defaultdict
import weaviate
from core.config import (
    WEAVIATE_URL, AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION,
    AZURE_DEPLOYMENT_NAME, AZURE_EMBEDDING_DEPLOYMENT_NAME,
    AZURE_EMBEDDING_API_BASE, AZURE_EMBEDDING_API_VERSION, AZURE_EMBEDDING_API_KEY,
    DEFAULT_MODULE, DEFAULT_SESSION_ID
)

logger = logging.getLogger(__name__)

# Global Weaviate client
weaviate_client: Optional[weaviate.WeaviateClient] = None

# In-memory fallback storage
in_memory_storage: Dict[str, List[Dict]] = defaultdict(list)

def initialize_weaviate() -> bool:
    """
    Initialize Weaviate client connection.
    """
    global weaviate_client
    try:
        if weaviate_client is not None:
            logger.info("Weaviate client already initialized.")
            return True
        
        weaviate_client = weaviate.connect_to_local()
        
        # Test the connection
        if weaviate_client.is_ready():
            version_info = weaviate_client.get_meta()
            logger.info(f"✓ Weaviate client initialized successfully - version: {version_info.get('version', 'Unknown')}")
            return True
        else:
            logger.error("Weaviate is not ready")
            return False
        
    except Exception as e:
        logger.error(f"❌ Weaviate initialization failed: {e}")
        return False

def _get_memory_key(tenant_id: str, session_id: Optional[str] = None, module: Optional[str] = None) -> str:
    """
    Generate a unique memory key for multi-tenant isolation.
    """
    session_id = session_id if session_id is not None else DEFAULT_SESSION_ID
    module = module if module is not None else DEFAULT_MODULE
    return f"{tenant_id}:{module}:{session_id}"

def store_memory(
    tenant_id: str,
    user_input: str,
    response_text: str,
    session_id: Optional[str] = None,
    module: Optional[str] = None,
    span: Optional[Any] = None
) -> None:
    """
    Store conversation memory in Weaviate.
    Falls back to in-memory storage if Weaviate fails.
    """
    mem_key = _get_memory_key(tenant_id, session_id, module)
    
    if not weaviate_client:
        logger.warning("Weaviate not available, using fallback storage")
        store_in_memory_fallback(tenant_id, user_input, response_text, session_id, module)
        return
    
    try:
        # Create memory object
        memory_data = {
            "user_id": mem_key,
            "user_input": user_input,
            "response_text": response_text,
            "timestamp": datetime.utcnow().isoformat(),
            "tenant_id": tenant_id,
            "module": module,
            "session_id": session_id
        }
        
        # Store in Weaviate
        collection = weaviate_client.collections.get("Mem0Memory")
        collection.data.insert(
            properties=memory_data
        )
        logger.info(f"✓ Stored memory in Weaviate for key: {mem_key}")
        
        if span:
            try:
                span.update(metadata={"weaviate_status": "stored", "weaviate_key": mem_key})
            except Exception:
                pass
                
    except Exception as e:
        logger.error(f"❌ Weaviate storage failed for key '{mem_key}': {e}")
        store_in_memory_fallback(tenant_id, user_input, response_text, session_id, module)
        if span:
            try:
                span.update(metadata={"weaviate_error": str(e)})
            except Exception:
                pass

def retrieve_memories(
    tenant_id: str,
    session_id: Optional[str] = None,
    module: Optional[str] = None,
    max_memories: int = 5
) -> List[Dict]:
    """
    Retrieve memories from Weaviate.
    Falls back to in-memory storage if Weaviate fails.
    """
    mem_key = _get_memory_key(tenant_id, session_id, module)
    
    if not weaviate_client:
        logger.warning("Weaviate not available, using fallback retrieval")
        return retrieve_from_memory_fallback(tenant_id, session_id, module, max_memories)
    
    try:
        # Query Weaviate for memories
        query = {
            "class": "Mem0Memory",
            "properties": ["user_input", "response_text", "timestamp", "tenant_id", "module", "session_id"],
            "where": {
                "path": ["user_id"],
                "operator": "Equal",
                "valueString": mem_key
            },
            "sort": [{"path": ["timestamp"], "order": "desc"}],
            "limit": max_memories
        }
        
        collection = weaviate_client.collections.get("Mem0Memory")
        result = collection.query.fetch_objects(
            limit=max_memories,
            filters=weaviate.classes.query.Filter.by_property("user_id").equal(mem_key)
        )
        
        memories = result.objects
        
        logger.info(f"✓ Retrieved {len(memories)} memories from Weaviate for key: {mem_key}")
        
        # Convert to standard format
        converted_memories = []
        for memory in memories:
            converted_memories.append({
                'user': memory.properties.get('user_input', ''),
                'bot': memory.properties.get('response_text', ''),
                'timestamp': memory.properties.get('timestamp', ''),
                'tenant_id': memory.properties.get('tenant_id', tenant_id),
                'module': memory.properties.get('module', module),
                'session_id': memory.properties.get('session_id', session_id)
            })
        
        if converted_memories:
            return converted_memories
        else:
            logger.info(f"No memories found in Weaviate for key: {mem_key}")
            return retrieve_from_memory_fallback(tenant_id, session_id, module, max_memories)
            
    except Exception as e:
        logger.error(f"❌ Weaviate retrieval failed for key '{mem_key}': {e}")
        return retrieve_from_memory_fallback(tenant_id, session_id, module, max_memories)

def close_weaviate() -> None:
    """
    Close Weaviate client connection.
    """
    global weaviate_client
    if weaviate_client:
        try:
            weaviate_client.close()
            weaviate_client = None
            logger.info("✓ Weaviate client connection closed")
        except Exception as e:
            logger.error(f"❌ Error during Weaviate shutdown: {e}")

def store_in_memory_fallback(
    tenant_id: str,
    user_input: str,
    response_text: str,
    session_id: Optional[str] = None,
    module: Optional[str] = None
) -> None:
    """
    Store memory in in-memory fallback storage.
    """
    mem_key = _get_memory_key(tenant_id, session_id, module)
    
    memory_data = {
        'user': user_input,
        'bot': response_text,
        'timestamp': datetime.utcnow().isoformat(),
        'tenant_id': tenant_id,
        'module': module,
        'session_id': session_id
    }
    
    in_memory_storage[mem_key].append(memory_data)
    logger.info(f"Stored memory in fallback storage for key: {mem_key}")

def retrieve_from_memory_fallback(
    tenant_id: str,
    session_id: Optional[str] = None,
    module: Optional[str] = None,
    max_memories: int = 5
) -> List[Dict]:
    """
    Retrieve memories from in-memory fallback storage.
    """
    mem_key = _get_memory_key(tenant_id, session_id, module)
    
    memories = in_memory_storage.get(mem_key, [])
    # Get the most recent memories
    recent_memories = memories[-max_memories:] if len(memories) > max_memories else memories
    
    logger.info(f"Retrieved {len(recent_memories)} memories from fallback storage for key: {mem_key}")
    return recent_memories

