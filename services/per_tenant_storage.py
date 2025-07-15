"""
Per-Tenant Storage Service

This module provides complete per-tenant isolation by creating separate collections
for each tenant for both vector storage (document chunks) and memory storage.
"""

import logging
from typing import Optional, List, Dict, Any
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from datetime import datetime
from collections import defaultdict
import re
import json

logger = logging.getLogger(__name__)

# Global client
weaviate_client: Optional[weaviate.WeaviateClient] = None


def initialize_per_tenant_storage() -> bool:
    """
    Initialize the per-tenant storage service.
    """
    global weaviate_client
    try:
        weaviate_client = weaviate.connect_to_local()
        if weaviate_client.is_ready():
            version_info = weaviate_client.get_meta()
            logger.info(f"Connected to Weaviate version: {version_info.get('version', 'Unknown')}")
            return True
        else:
            logger.error("Weaviate is not ready")
            return False
    except Exception as e:
        logger.error(f"Weaviate connection failed: {e}")
        return False

def _sanitize_tenant_id(tenant_id: str) -> str:
    """
    Sanitize tenant ID for use in collection names.
    Weaviate collection names must be alphanumeric and start with a letter.
    """
    # Replace invalid characters with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', tenant_id)
    # Ensure it starts with a letter
    if sanitized and not sanitized[0].isalpha():
        sanitized = 't_' + sanitized
    return sanitized

def _get_collection_names(tenant_id: str) -> tuple[str, str]:
    """
    Get collection names for a tenant.
    Returns (vector_collection_name, memory_collection_name)
    """
    sanitized_tenant = _sanitize_tenant_id(tenant_id)
    vector_collection = f"Documents_{sanitized_tenant}"
    memory_collection = f"Memory_{sanitized_tenant}"
    return vector_collection, memory_collection

def ensure_tenant_collections(tenant_id: str) -> bool:
    """
    Ensure that both vector and memory collections exist for a tenant.
    """
    if not weaviate_client:
        logger.error("Weaviate client not initialized")
        return False
    
    vector_collection, memory_collection = _get_collection_names(tenant_id)
    
    try:
        # Create vector collection if it doesn't exist
        if not weaviate_client.collections.exists(vector_collection):
            logger.info(f"Creating vector collection: {vector_collection}")
            weaviate_client.collections.create(
                name=vector_collection,
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="metadata", data_type=DataType.TEXT),
                    Property(name="tenant_id", data_type=DataType.TEXT),
                    Property(name="module", data_type=DataType.TEXT),
                    Property(name="created_at", data_type=DataType.DATE),
                ],
                vectorizer_config=None  # No vectorizer - we'll add vectors manually
            )
            logger.info(f"✓ Created vector collection: {vector_collection}")
        
        # Create memory collection if it doesn't exist
        if not weaviate_client.collections.exists(memory_collection):
            logger.info(f"Creating memory collection: {memory_collection}")
            weaviate_client.collections.create(
                name=memory_collection,
                properties=[
                    Property(name="user_input", data_type=DataType.TEXT),
                    Property(name="assistant_response", data_type=DataType.TEXT),
                    Property(name="session_id", data_type=DataType.TEXT),
                    Property(name="module", data_type=DataType.TEXT),
                    Property(name="created_at", data_type=DataType.DATE),
                ],
                vectorizer_config=None  # No vectorization for memory
            )
            logger.info(f"✓ Created memory collection: {memory_collection}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to ensure collections for tenant {tenant_id}: {e}")
        return False

def store_document_chunks(
    tenant_id: str,
    chunks: List[Dict],
    module: Optional[str] = None
) -> bool:
    """
    Store document chunks in tenant-specific vector collection.
    """
    if not weaviate_client:
        logger.error("Weaviate client not initialized")
        return False
    
    vector_collection, _ = _get_collection_names(tenant_id)
    
    try:
        # Ensure collections exist
        if not ensure_tenant_collections(tenant_id):
            return False
        
        collection = weaviate_client.collections.get(vector_collection)
        
        # Import embedding function
        from utils.embeddings import get_openai_embedding
        
        # Prepare data for insertion with vectors
        for chunk in chunks:
            # Get embedding for the content
            content = chunk.get('content', '')
            
            # Debug: Log the content being processed (minimal logging)
            logger.debug(f"Processing chunk with content length: {len(content) if content else 0}")
            
            vector = get_openai_embedding(content)
            
            if not vector:
                logger.warning(f"Failed to get embedding for chunk, skipping")
                continue
            
            # Insert with vector
            collection.data.insert(
                properties={
                    'content': content,
                    'metadata': str(chunk.get('metadata', {})),
                    'tenant_id': tenant_id,
                    'module': module or 'general',
                    'created_at': datetime.utcnow().isoformat() + 'Z'
                },
                vector=vector
            )
        
        logger.info(f"✓ Stored {len(chunks)} chunks in {vector_collection}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to store document chunks for tenant {tenant_id}: {e}")
        return False

def retrieve_document_chunks(
    tenant_id: str,
    query: str,
    module: Optional[str] = None,
    top_k: int = 3
) -> List[str]:
    """
    Retrieve document chunks from tenant-specific vector collection.
    """
    if not weaviate_client:
        logger.error("Weaviate client not initialized")
        return []
    
    vector_collection, _ = _get_collection_names(tenant_id)
    
    try:
        # Ensure collections exist
        if not ensure_tenant_collections(tenant_id):
            return []
        
        collection = weaviate_client.collections.get(vector_collection)
        
        # Import embedding function for query
        from utils.embeddings import get_openai_embedding
        
        # Get embedding for the query
        query_vector = get_openai_embedding(query)
        if not query_vector:
            logger.error("Failed to generate embedding for query")
            return []
        
        # Perform vector search
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_properties=["content", "metadata", "module", "tenant_id"]
        )
        
        results = []
        for obj in response.objects:
            # Since we're using per-tenant collections, all results are for this tenant
            # Just filter by module if specified
            if module is None or obj.properties.get('module') == module:
                content = obj.properties.get('content', '')
                metadata = obj.properties.get('metadata', '')
                
                # Try to extract document source from metadata
                document_source = "Unknown"
                if metadata and isinstance(metadata, str):
                    try:
                        import json
                        metadata_dict = json.loads(metadata)
                        if 'source' in metadata_dict:
                            document_source = metadata_dict['source']
                    except:
                        pass
                
                # Add document source to content for context
                if content:
                    results.append(f"[Source: {document_source}]\n{content}")
        
        logger.info(f"✓ Retrieved {len(results)} chunks from {vector_collection}")
        return results
        
    except Exception as e:
        logger.error(f"Failed to retrieve document chunks for tenant {tenant_id}: {e}")
        return []

def store_memory(
    tenant_id: str,
    user_input: str,
    response_text: str,
    session_id: str,  # required, used as user_id
    module: Optional[str] = None,
    span: Optional[Any] = None
) -> None:
    """
    Store a conversation turn using Weaviate for tenant/module/session isolation.
    """
    if not weaviate_client:
        logger.error("Weaviate client not initialized")
        return
    
    _, memory_collection = _get_collection_names(tenant_id)
    
    try:
        # Ensure collections exist
        if not ensure_tenant_collections(tenant_id):
            return
        
        collection = weaviate_client.collections.get(memory_collection)
        
        # Create memory data
        memory_data = {
            "user_input": user_input,
            "assistant_response": response_text,
            "session_id": session_id,
            "module": module or "general",
            "created_at": datetime.utcnow().isoformat() + 'Z'
        }
        
        # Store in Weaviate
        collection.data.insert(
            properties=memory_data
        )
        
        logger.info(f"✅ Stored memory for {tenant_id}:{session_id}:{module} in Weaviate")
        
        if span:
            try:
                span.update(metadata={"memory_status": "Stored"})
            except Exception:
                pass
                
    except Exception as e:
        logger.error(f"❌ Failed to store memory for {tenant_id}:{session_id}:{module} in Weaviate: {e}")
        if span:
            try:
                span.update(metadata={"memory_status": "Failed"})
            except Exception:
                pass

# Example streaming handler (pseudo-code)
def handle_streaming_response(tenant_id, user_input, session_id, module, stream_chunks):
    """
    Buffer streaming LLM response and store only the final message.
    """
    full_response = ""
    for chunk in stream_chunks:
        full_response += chunk
    store_memory(
        tenant_id=tenant_id,
        user_input=user_input,
        response_text=full_response,
        session_id=session_id,
        module=module
    )


def retrieve_memories(
    tenant_id: str,
    session_id: str,  # required, used as user_id
    module: Optional[str] = None,
    max_memories: int = 5
) -> List[Dict]:
    """
    Retrieve recent memories for a specific tenant, session, and module using Weaviate.
    """
    if not weaviate_client:
        logger.error("Weaviate client not initialized")
        return []
    
    _, memory_collection = _get_collection_names(tenant_id)
    
    try:
        # Ensure collections exist
        if not ensure_tenant_collections(tenant_id):
            return []
        
        collection = weaviate_client.collections.get(memory_collection)
        
        # Create filters for session_id and module
        filters = weaviate.classes.query.Filter.by_property("session_id").equal(session_id)
        
        if module:
            # Add module filter
            module_filter = weaviate.classes.query.Filter.by_property("module").equal(module)
            filters = filters & module_filter
        
        # Get memories ordered by creation date (newest first)
        response = collection.query.fetch_objects(
            limit=max_memories,
            filters=filters,
            sort=weaviate.classes.query.Sort.by_property("created_at", ascending=False),
            return_properties=["user_input", "assistant_response", "session_id", "module", "created_at"]
        )
        
        memories = []
        for obj in response.objects:
            memories.append({
                "user": obj.properties.get("user_input", ""),
                "bot": obj.properties.get("assistant_response", ""),
                "session_id": obj.properties.get("session_id", ""),
                "module": obj.properties.get("module", ""),
                "created_at": obj.properties.get("created_at", "")
            })
        
        logger.info(f"✅ Retrieved {len(memories)} memories for {tenant_id}:{session_id}:{module} from Weaviate")
        return memories
        
    except Exception as e:
        logger.error(f"❌ Failed to retrieve memories for {tenant_id}:{session_id}:{module} from Weaviate: {e}")
        return []

def list_tenant_collections() -> List[str]:
    """
    List all tenant collections in Weaviate.
    """
    if not weaviate_client:
        return []
    
    try:
        collections = weaviate_client.collections.list_all()
        tenant_collections = []
        
        for collection in collections:
            # Handle both string and object representations
            if isinstance(collection, str):
                collection_name = collection
            else:
                collection_name = getattr(collection, 'name', str(collection))
            
            if collection_name.startswith(('Documents_', 'Memory_')):
                tenant_collections.append(collection_name)
        
        return tenant_collections
    except Exception as e:
        logger.error(f"Failed to list tenant collections: {e}")
        return []

def delete_tenant_data(tenant_id: str) -> bool:
    """
    Delete all data for a specific tenant.
    """
    if not weaviate_client:
        logger.error("Weaviate client not initialized")
        return False
    
    vector_collection, memory_collection = _get_collection_names(tenant_id)
    
    try:
        # Delete vector collection
        if weaviate_client.collections.exists(vector_collection):
            weaviate_client.collections.delete(vector_collection)
            logger.info(f"✓ Deleted vector collection: {vector_collection}")
        
        # Delete memory collection
        if weaviate_client.collections.exists(memory_collection):
            weaviate_client.collections.delete(memory_collection)
            logger.info(f"✓ Deleted memory collection: {memory_collection}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to delete tenant data for {tenant_id}: {e}")
        return False

def close_per_tenant_storage() -> None:
    """
    Properly close the per-tenant storage service.
    """
    global weaviate_client
    if weaviate_client:
        try:
            weaviate_client.close()
            weaviate_client = None
            logger.info("Per-tenant storage service closed.")
        except Exception as e:
            logger.error(f"Error during per-tenant storage shutdown: {e}")
    else:
        logger.info("Per-tenant storage service was not initialized.") 