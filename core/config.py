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

# Azure OpenAI Embedding Model Settings
AZURE_EMBEDDING_API_KEY = get_env_var("AZURE_EMBEDDING_API_KEY")
AZURE_EMBEDDING_API_BASE = get_env_var("AZURE_EMBEDDING_API_BASE")
AZURE_EMBEDDING_API_VERSION = get_env_var("AZURE_EMBEDDING_API_VERSION", required=False, default="2023-05-15")
AZURE_EMBEDDING_DEPLOYMENT_NAME = get_env_var("AZURE_EMBEDDING_DEPLOYMENT_NAME")


# Weaviate Settings
WEAVIATE_URL = get_env_var("WEAVIATE_URL", required=False, default="http://localhost:8080")
WEAVIATE_COLLECTION_NAME = get_env_var("WEAVIATE_COLLECTION_NAME", required=False, default="multi_tenant_collection")

# Mem0 Settings (optional)
MEM0_CLUSTER_URL = get_env_var("MEM0_CLUSTER_URL", required=False, default="http://localhost:8080")
MEM0_COLLECTION_NAME = get_env_var("MEM0_COLLECTION_NAME", required=False, default="Mem0Memory")

# Other configurable constants
DEFAULT_CHUNK_SIZE = int(get_env_var("DEFAULT_CHUNK_SIZE", required=False, default="500"))
DEFAULT_CHUNK_OVERLAP = int(get_env_var("DEFAULT_CHUNK_OVERLAP", required=False, default="50"))

# Add any other config variables here as needed

DEFAULT_MODULE = "others"

DELETE_COLLECTION_ON_INGEST = False