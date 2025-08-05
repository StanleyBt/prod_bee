"""
Mem0 Service for User Preference Management

This service provides simple functions to store and retrieve user preferences
for context injection in prompts.
"""

import logging
from typing import Optional
from mem0 import Memory
from core.config import (
    AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION,
    AZURE_DEPLOYMENT_NAME, AZURE_EMBEDDING_DEPLOYMENT_NAME,
    MEM0_COLLECTION_NAME,
    AZURE_EMBEDDING_API_KEY, AZURE_EMBEDDING_API_VERSION,
    AZURE_EMBEDDING_API_BASE,
    DEFAULT_MODULE, DEFAULT_SESSION_ID
)

import os
import atexit
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Global Mem0 client
mem0_client: Optional[Memory] = None

def _ensure_mem0_cleanup():
    """
    Ensure Mem0 cleanup happens even if not explicitly called.
    """
    global mem0_client
    if mem0_client:
        mem0_client = None

# Register cleanup function to run at exit
atexit.register(_ensure_mem0_cleanup)

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
        
        global mem0_client
        mem0_client = Memory.from_config(config)
        logger.info("Mem0 service initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Mem0 service: {e}")
        return False

def _get_user_id(tenant_id: str, user_id: str, session_id: Optional[str] = None, module: Optional[str] = None, role: Optional[str] = None) -> str:
    """
    Generate a unique user ID for Mem0.
    """
    session = session_id or DEFAULT_SESSION_ID
    mod = module or DEFAULT_MODULE
    user_role = role or "general"
    
    # Create a simple, predictable ID that avoids special characters
    clean_tenant = ''.join(c for c in tenant_id if c.isalnum())[:8]
    clean_user = ''.join(c for c in user_id if c.isalnum())[:8]
    clean_module = ''.join(c for c in mod if c.isalnum())[:8]
    clean_session = ''.join(c for c in session if c.isalnum())[:8]
    clean_role = ''.join(c for c in user_role if c.isalnum())[:8]
    
    return f"{clean_tenant}_{clean_user}_{clean_module}_{clean_role}_{clean_session}"



def get_user_context(
    tenant_id: str,
    user_id: str,
    session_id: Optional[str] = None,
    module: Optional[str] = None,
    role: Optional[str] = None
) -> str:
    """
    Get user context from memories for prompt injection.
    """
    if not mem0_client:
        logger.error("Mem0 client not initialized")
        return ""
    
    try:
        mem0_user_id = _get_user_id(tenant_id, user_id, session_id, module, role)
        result = mem0_client.get_all(user_id=mem0_user_id)
        if not result or not result.get('results'):
            return ""
        
        # Let Mem0's natural language capabilities handle context extraction
        context_parts = []
        for memory in result.get('results', []):
            context_parts.append(memory.get('memory', ''))
        
        return " ".join(context_parts)
    except Exception as e:
        logger.error(f"Failed to retrieve user context: {e}")
        return ""

def store_user_memory(
    tenant_id: str,
    user_id: str,
    user_input: str,
    response_text: str,
    session_id: Optional[str] = None,
    module: Optional[str] = None,
    role: Optional[str] = None
) -> bool:
    """
    Store user interaction in Mem0 for future context retrieval.
    """
    if not mem0_client:
        logger.error("Mem0 client not initialized")
        return False
    
    try:
        mem0_user_id = _get_user_id(tenant_id, user_id, session_id, module, role)
        
        # Create a natural language memory entry
        memory_text = f"User asked: {user_input}. Assistant responded: {response_text}"
        
        # Store in Mem0
        result = mem0_client.add(
            memory_text,
            user_id=mem0_user_id
        )
        
        if result:
            logger.info(f"Stored user memory for {mem0_user_id}")
            return True
        else:
            logger.warning(f"Mem0 storage returned empty result for {mem0_user_id}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to store user memory: {e}")
        return False

def close_mem0_service():
    """
    Close Mem0 service connections.
    """
    global mem0_client
    try:
        if mem0_client:
            # Mem0 doesn't have an explicit close method, but we can clear the reference
            # This should trigger garbage collection and close underlying connections
            mem0_client = None
            logger.info("Mem0 service connections closed")
        else:
            logger.info("Mem0 service was not initialized")
    except Exception as e:
        logger.error(f"Error closing Mem0 service: {e}")
        # Ensure client is cleared even if cleanup fails
        mem0_client = None