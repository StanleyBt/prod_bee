"""
Shared embedding utilities for the RAG system with intelligent caching.
"""

import logging
import os
from typing import Optional, List
import openai
from core.config import (
    AZURE_EMBEDDING_API_KEY,
    AZURE_EMBEDDING_API_BASE,
    AZURE_EMBEDDING_API_VERSION,
    AZURE_EMBEDDING_DEPLOYMENT_NAME
)

logger = logging.getLogger(__name__)

# Global OpenAI client for reuse
_openai_client: Optional[openai.AzureOpenAI] = None

def _get_openai_client() -> Optional[openai.AzureOpenAI]:
    """Get or create the OpenAI client."""
    global _openai_client
    
    if _openai_client is None:
        try:
            if not AZURE_EMBEDDING_API_KEY or not AZURE_EMBEDDING_API_BASE or not AZURE_EMBEDDING_DEPLOYMENT_NAME:
                logger.error("Azure embedding configuration is missing")
                return None
            
            _openai_client = openai.AzureOpenAI(
                api_key=AZURE_EMBEDDING_API_KEY,
                api_version=AZURE_EMBEDDING_API_VERSION,
                azure_endpoint=AZURE_EMBEDDING_API_BASE,
            )
            logger.debug("✅ OpenAI client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            return None
    
    return _openai_client

def get_openai_embedding(text: str, use_cache: bool = True) -> Optional[List[float]]:
    """
    Get OpenAI embedding for the given text using Azure OpenAI with intelligent caching.
    
    Args:
        text: The text to embed
        use_cache: Whether to use caching (default: True)
        
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
        
        # Check cache first if enabled
        if use_cache:
            try:
                from utils.embedding_cache import get_embedding_cache
                cache = get_embedding_cache()
                cached_embedding = cache.get(cleaned_text)
                
                if cached_embedding is not None:
                    logger.debug(f"✅ Cache hit for embedding: {cleaned_text[:50]}...")
                    return cached_embedding
                
                logger.debug(f"Cache miss for embedding: {cleaned_text[:50]}...")
                
            except Exception as e:
                logger.warning(f"Cache error, falling back to API: {e}")
        
        # Get OpenAI client
        client = _get_openai_client()
        if not client:
            return None
        
        # Make API call
        logger.debug(f"Making API call for embedding: {cleaned_text[:50]}...")
        model_name = AZURE_EMBEDDING_DEPLOYMENT_NAME
        
        response = client.embeddings.create(
            input=cleaned_text,
            model=model_name
        )
        
        embedding = response.data[0].embedding
        
        # Store in cache if enabled
        if use_cache and embedding:
            try:
                from utils.embedding_cache import get_embedding_cache
                cache = get_embedding_cache()
                cache.put(cleaned_text, embedding)
                logger.debug(f"✅ Cached embedding: {cleaned_text[:50]}...")
            except Exception as e:
                logger.warning(f"Failed to cache embedding: {e}")
        
        return embedding
        
    except Exception as e:
        logger.error(f"Embedding call failed: {e}")
        return None

def get_embedding_cache_stats() -> Optional[dict]:
    """Get embedding cache statistics."""
    try:
        from utils.embedding_cache import get_embedding_cache
        cache = get_embedding_cache()
        return cache.get_stats()
    except Exception as e:
        logger.warning(f"Failed to get cache stats: {e}")
        return None

def clear_embedding_cache():
    """Clear the embedding cache."""
    try:
        from utils.embedding_cache import get_embedding_cache
        cache = get_embedding_cache()
        cache.clear()
        logger.info("✅ Embedding cache cleared")
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")

def shutdown_embedding_cache():
    """Shutdown the embedding cache."""
    try:
        from utils.embedding_cache import shutdown_embedding_cache
        shutdown_embedding_cache()
        logger.info("✅ Embedding cache shutdown")
    except Exception as e:
        logger.error(f"Failed to shutdown cache: {e}") 