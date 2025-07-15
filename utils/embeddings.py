"""
Shared embedding utilities for the RAG system.
"""

import logging
from typing import Optional, List
import openai
from core.config import (
    AZURE_EMBEDDING_API_KEY,
    AZURE_EMBEDDING_API_BASE,
    AZURE_EMBEDDING_API_VERSION,
    AZURE_EMBEDDING_DEPLOYMENT_NAME
)

logger = logging.getLogger(__name__)

def get_openai_embedding(text: str) -> Optional[List[float]]:
    """
    Get OpenAI embedding for the given text using Azure OpenAI.
    
    Args:
        text: The text to embed
        
    Returns:
        List of floats representing the embedding, or None if failed
    """
    try:
        # Validate input text
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None
            
        # Clean the text - remove any problematic characters
        cleaned_text = text.strip()
        if len(cleaned_text) == 0:
            logger.warning("Text is empty after cleaning")
            return None
            
        if not AZURE_EMBEDDING_API_KEY or not AZURE_EMBEDDING_API_BASE or not AZURE_EMBEDDING_DEPLOYMENT_NAME:
            logger.error("Azure embedding configuration is missing")
            return None

        client = openai.AzureOpenAI(
            api_key=AZURE_EMBEDDING_API_KEY,
            api_version=AZURE_EMBEDDING_API_VERSION,
            azure_endpoint=AZURE_EMBEDDING_API_BASE,
        )
        
        # Log the text being embedded for debugging
        logger.debug(f"Embedding text (first 100 chars): {cleaned_text[:100]}...")
        
        # For Azure OpenAI, we might need to use the deployment name differently
        # Try using the full model name instead of deployment name
        model_name = AZURE_EMBEDDING_DEPLOYMENT_NAME
        
        response = client.embeddings.create(
            input=cleaned_text,  # Pass as string, not list
            model=model_name
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding call failed: {e}")
        return None 