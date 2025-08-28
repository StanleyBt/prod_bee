"""
Per-Tenant Storage Service

This module provides complete per-tenant isolation by creating separate collections
for each tenant for both vector storage (document chunks) and memory storage.
"""

import logging
from typing import Optional, List, Dict, Any
import weaviate
from weaviate.classes.config import Property, DataType
from datetime import datetime, timezone
import re
import atexit
from core.config import WEAVIATE_URL, WEAVIATE_API_KEY

logger = logging.getLogger(__name__)

# Global client
weaviate_client: Optional[weaviate.WeaviateClient] = None

def _ensure_cleanup():
    """
    Ensure cleanup happens even if not explicitly called.
    This is a safety measure for garbage collection.
    """
    global weaviate_client
    if weaviate_client:
        try:
            weaviate_client.close()
        except:
            pass
        weaviate_client = None

# Register cleanup function to run at exit
atexit.register(_ensure_cleanup)


def initialize_per_tenant_storage() -> bool:
    """
    Initialize the per-tenant storage service.
    """
    global weaviate_client
    try:
        # Close any existing connection first
        if weaviate_client:
            try:
                weaviate_client.close()
            except:
                pass
            weaviate_client = None
        
        weaviate_client = weaviate.connect_to_local()
        if weaviate_client.is_ready():
            version_info = weaviate_client.get_meta()
            logger.info(f"Connected to Weaviate version: {version_info.get('version', 'Unknown')}")
            return True
        else:
            logger.error("Weaviate is not ready")
            if weaviate_client:
                weaviate_client.close()
                weaviate_client = None
            return False
    except Exception as e:
        logger.error(f"Weaviate connection failed: {e}")
        if weaviate_client:
            try:
                weaviate_client.close()
            except:
                pass
            weaviate_client = None
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
                    Property(name="role", data_type=DataType.TEXT),
                    Property(name="document_name", data_type=DataType.TEXT),
                    Property(name="filename", data_type=DataType.TEXT),
                    Property(name="page_count", data_type=DataType.INT),
                    Property(name="word_count", data_type=DataType.INT),
                    Property(name="chunk_index", data_type=DataType.INT),
                    Property(name="total_chunks", data_type=DataType.INT),
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
                    Property(name="role", data_type=DataType.TEXT),
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
    module: Optional[str] = None,
    role: Optional[str] = None,
    video_url: Optional[str] = None  # NEW: Optional video URL parameter
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
            
            # Get metadata from chunk
            chunk_metadata = chunk.get('metadata', {})
            
            # Add video URL to metadata if provided
            if video_url:
                chunk_metadata['video_url'] = video_url
            
            # Insert with vector and enhanced metadata
            collection.data.insert(
                properties={
                    'content': content,
                    'metadata': str(chunk_metadata),
                    'tenant_id': tenant_id,
                    'module': module or 'general',
                    'role': role or 'general',
                    'document_name': chunk_metadata.get('document_name', ''),
                    'filename': chunk_metadata.get('filename', ''),
                    'page_count': chunk_metadata.get('page_count', 0),
                    'word_count': chunk_metadata.get('word_count', 0),
                    'chunk_index': chunk_metadata.get('chunk_index', 0),
                    'total_chunks': chunk_metadata.get('total_chunks', 1),
                    'created_at': datetime.now(timezone.utc).isoformat(),
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
    role: Optional[str] = None,
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
            return_properties=["content", "metadata", "module", "role", "tenant_id", "document_name", "filename", "page_count", "word_count", "chunk_index", "total_chunks"]
        )
        
        results = []
        for obj in response.objects:
            # Since we're using per-tenant collections, all results are for this tenant
            # Filter by module and role if specified
            obj_module = obj.properties.get('module')
            obj_role = obj.properties.get('role')
            
            if (module is None or obj_module.lower() == module.lower()) and (role is None or obj_role.lower() == role.lower()):
                content = obj.properties.get('content', '')
                metadata = obj.properties.get('metadata', '')
                
                # Extract video URL from metadata
                video_url = ''
                try:
                    import ast
                    metadata_dict = ast.literal_eval(metadata) if metadata else {}
                    video_url = metadata_dict.get('video_url', '')
                except:
                    pass
                
                # Extract document information
                document_name = obj.properties.get('document_name', 'Unknown')
                filename = obj.properties.get('filename', 'Unknown')
                page_count = obj.properties.get('page_count', 0)
                word_count = obj.properties.get('word_count', 0)
                chunk_index = obj.properties.get('chunk_index', 0)
                total_chunks = obj.properties.get('total_chunks', 1)
                
                # Create enhanced context with video URL
                if total_chunks == 1:
                    context = f"[Document: {document_name} | File: {filename} | Pages: {page_count} | Words: {word_count}"
                else:
                    context = f"[Document: {document_name} | File: {filename} | Chunk {chunk_index + 1}/{total_chunks} | Pages: {page_count} | Words: {word_count}"
                
                # Add video URL to context if available
                if video_url:
                    context += f" | Video: {video_url}"
                
                context += "]"
                
                # Add document context to content
                if content:
                    results.append(f"{context}\n{content}")
        
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
    role: Optional[str] = None,
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
            "role": role or "general",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Store in Weaviate
        collection.data.insert(
            properties=memory_data
        )
        
        logger.info(f"✅ Stored memory for {tenant_id}:{session_id}:{module}:{role} in Weaviate")
        
        if span:
            try:
                span.update(metadata={"memory_status": "Stored"})
            except Exception:
                pass
                
    except Exception as e:
        logger.error(f"❌ Failed to store memory for {tenant_id}:{session_id}:{module}:{role} in Weaviate: {e}")
        if span:
            try:
                span.update(metadata={"memory_status": "Failed"})
            except Exception:
                pass


def retrieve_memories(
    tenant_id: str,
    session_id: str,  # required, used as user_id
    module: Optional[str] = None,
    role: Optional[str] = None,
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
        
        # Create filters for session_id, module, and role
        filters = weaviate.classes.query.Filter.by_property("session_id").equal(session_id)
        
        if module:
            # Add module filter (Weaviate stores exact case, so we match exactly)
            module_filter = weaviate.classes.query.Filter.by_property("module").equal(module)
            filters = filters & module_filter
        
        if role:
            # Add role filter (Weaviate stores exact case, so we match exactly)
            role_filter = weaviate.classes.query.Filter.by_property("role").equal(role)
            filters = filters & role_filter
        
        # Get memories ordered by creation date (newest first)
        response = collection.query.fetch_objects(
            limit=max_memories,
            filters=filters,
            sort=weaviate.classes.query.Sort.by_property("created_at", ascending=False),
            return_properties=["user_input", "assistant_response", "session_id", "module", "role", "created_at"]
        )
        
        memories = []
        for obj in response.objects:
            memories.append({
                "user": obj.properties.get("user_input", ""),
                "bot": obj.properties.get("assistant_response", ""),
                "session_id": obj.properties.get("session_id", ""),
                "module": obj.properties.get("module", ""),
                "role": obj.properties.get("role", ""),
                "created_at": obj.properties.get("created_at", "")
            })
        
        logger.info(f"✅ Retrieved {len(memories)} memories for {tenant_id}:{session_id}:{module}:{role} from Weaviate")
        return memories
        
    except Exception as e:
        logger.error(f"❌ Failed to retrieve memories for {tenant_id}:{session_id}:{module}:{role} from Weaviate: {e}")
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



def clear_session_conversations(tenant_id: str, session_id: str) -> dict:
    """
    Clear all Weaviate conversation memories for a specific session.
    Preserves Mem0 memories - only cleans up Weaviate conversation history.
    Returns a dict with status and count of deleted messages.
    """
    if not weaviate_client:
        return {"success": False, "error": "Weaviate client not initialized", "deleted_count": 0}
    
    try:
        deleted_count = 0
        _, memory_collection = _get_collection_names(tenant_id)
        
        # Ensure collection exists
        if not ensure_tenant_collections(tenant_id):
            return {"success": False, "error": f"Failed to ensure collections exist for tenant {tenant_id}", "deleted_count": 0}
        
        collection = weaviate_client.collections.get(memory_collection)
        
        # Get all objects for the session
        response = collection.query.fetch_objects(
            limit=1000,
            return_properties=["session_id", "user_input", "created_at"]
        )
        
        # Filter by session_id
        for obj in response.objects:
            obj_session_id = obj.properties.get('session_id', '')
            
            # Check if this object matches our session
            if obj_session_id == session_id:
                try:
                    collection.data.delete_by_id(obj.uuid)
                    deleted_count += 1
                    user_input = obj.properties.get('user_input', '')[:50]
                    logger.info(f"🗑️ Deleted conversation: session={session_id}, user='{user_input}...'")
                except Exception as e:
                    logger.warning(f"Failed to delete conversation {obj.uuid}: {e}")
        
        if deleted_count > 0:
            logger.info(f"✅ Cleared {deleted_count} Weaviate conversations for session {session_id} (Mem0 memories preserved)")
            return {"success": True, "deleted_count": deleted_count, "message": f"Cleared {deleted_count} conversations"}
        else:
            logger.info(f"ℹ️ No Weaviate conversations found for session {session_id}")
            return {"success": True, "deleted_count": 0, "message": "No conversations found to clear"}
        
    except Exception as e:
        error_msg = f"Failed to clear conversations for {tenant_id}:{session_id}: {e}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg, "deleted_count": 0}

def close_per_tenant_storage() -> None:
    """
    Properly close the per-tenant storage service.
    """
    global weaviate_client
    if weaviate_client:
        try:
            # Force close any remaining connections
            weaviate_client.close()
            # Clear the client reference
            weaviate_client = None
            logger.info("Per-tenant storage service closed.")
        except Exception as e:
            logger.error(f"Error during per-tenant storage shutdown: {e}")
            # Ensure client is cleared even if close fails
            weaviate_client = None
    else:
        logger.info("Per-tenant storage service was not initialized.") 