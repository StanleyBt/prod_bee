# core/config.py

import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()  # Load from .env file if present

def get_env_var(key: str, required: bool = True, default: Optional[str] = None) -> Optional[str]:
    """Utility to get environment variables with optional default and required flag."""
    value = os.getenv(key, default)
    if required and value is None:
        raise EnvironmentError(f"Required environment variable '{key}' not set.")
    return value

# Azure OpenAI Chat Model Settings
AZURE_API_KEY = get_env_var("AZURE_OPENAI_API_KEY")
AZURE_API_BASE = get_env_var("AZURE_OPENAI_API_BASE")
AZURE_API_VERSION = get_env_var("AZURE_API_VERSION", required=False, default="2025-01-01-preview")
AZURE_DEPLOYMENT_NAME = get_env_var("AZURE_OPENAI_DEPLOYMENT_NAME")

# Azure OpenAI Embedding Settings
AZURE_EMBEDDING_API_KEY = get_env_var("AZURE_EMBEDDING_API_KEY")
AZURE_EMBEDDING_API_BASE = get_env_var("AZURE_EMBEDDING_API_BASE")
AZURE_EMBEDDING_API_VERSION = get_env_var("AZURE_EMBEDDING_API_VERSION", required=False, default="2025-01-01-preview")
AZURE_EMBEDDING_DEPLOYMENT_NAME = get_env_var("AZURE_EMBEDDING_DEPLOYMENT_NAME")

# Weaviate Settings
WEAVIATE_URL = get_env_var("WEAVIATE_URL", required=False, default="http://localhost:8080")
WEAVIATE_API_KEY = get_env_var("WEAVIATE_API_KEY", required=False, default=None)

# Mem0 Settings
MEM0_COLLECTION_NAME = get_env_var("MEM0_COLLECTION_NAME", required=False, default="Mem0Memory")

# Default Values
DEFAULT_MODULE = "default"
DEFAULT_SESSION_ID = "default_session"

# Document Processing Settings
# Original single chunk per document logic for user manuals
DEFAULT_CHUNK_SIZE = 50000   # Large chunk size for single chunk per document
DEFAULT_CHUNK_OVERLAP = 0    # No overlap since we want single chunks
SINGLE_CHUNK_PER_DOCUMENT = True  # Original: single chunk per document

# Context limits for RAG
MAX_CONTEXT_CHARS = 30000    # Maximum context length in characters (roughly 7500 tokens)
MAX_CHUNKS_PER_QUERY = 3     # Maximum number of chunks to retrieve per query