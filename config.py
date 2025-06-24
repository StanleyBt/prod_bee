# config.py

import os
from dotenv import load_dotenv
from typing import Optional, overload

load_dotenv() 

@overload
def get_env_var(key: str, required: bool = True, default: None = None) -> str: ...

@overload  
def get_env_var(key: str, required: bool = False, default: Optional[str] = None) -> Optional[str]: ...

def get_env_var(key: str, required: bool = True, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(key, default)
    if required and value is None:
        raise EnvironmentError(f"Required environment variable {key} not set.")
    return value

# Azure Chat Model Settings
AZURE_API_KEY = get_env_var("AZURE_OPENAI_API_KEY")
AZURE_API_BASE = get_env_var("AZURE_OPENAI_API_BASE")
AZURE_API_VERSION = get_env_var("AZURE_API_VERSION", required=False, default="2025-01-01-preview")
AZURE_DEPLOYMENT_NAME = get_env_var("AZURE_OPENAI_DEPLOYMENT_NAME")

# Azure Embedding Model Settings
AZURE_EMBEDDING_API_KEY = get_env_var("AZURE_EMBEDDING_API_KEY")
AZURE_EMBEDDING_API_BASE = get_env_var("AZURE_EMBEDDING_API_BASE")
AZURE_EMBEDDING_API_VERSION = get_env_var("AZURE_EMBEDDING_API_VERSION", required=False, default="2023-05-15")
AZURE_EMBEDDING_DEPLOYMENT_NAME = get_env_var("AZURE_EMBEDDING_DEPLOYMENT_NAME")

# Add any other config variables here as needed (e.g., Mem0/Weaviate endpoints)
