# llm.py

import logging
import litellm
from core.config import (
    AZURE_API_KEY,
    AZURE_API_BASE,
    AZURE_API_VERSION,
    AZURE_DEPLOYMENT_NAME,
)

logger = logging.getLogger(__name__)

def test_llm_connection():
    """Test Azure OpenAI LLM connection and credentials."""
    try:
        resp = litellm.completion(
            model=f"azure/{AZURE_DEPLOYMENT_NAME}",
            messages=[{"role": "user", "content": "Hello AI"}],
            api_key=AZURE_API_KEY,
            api_base=AZURE_API_BASE,
            api_version=AZURE_API_VERSION,
        )
        # Access response content safely
        try:
            content = resp.choices[0].message.content
        except (AttributeError, IndexError):
            content = str(resp)
        logger.info(f"LLM response: {content}")
        return True
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return False

def generate_llm_response(prompt: str, span=None):
    """Generate a response from Azure OpenAI LLM via litellm."""
    try:
        resp = litellm.completion(
            model=f"azure/{AZURE_DEPLOYMENT_NAME}",
            messages=[{"role": "user", "content": prompt}],
            api_key=AZURE_API_KEY,
            api_base=AZURE_API_BASE,
            api_version=AZURE_API_VERSION,
        )
        
        # Extract response content safely
        try:
            response_text = resp.choices[0].message.content
        except (AttributeError, IndexError):
            # Fallback to string conversion
            response_text = str(resp)
        
        # Update span with basic metrics if available
        if span:
            try:
                span.update(output={
                    "model": AZURE_DEPLOYMENT_NAME,
                    "response_length": len(response_text) if response_text else 0,
                    "provider": "azure_openai"
                })
            except Exception as e:
                logger.warning(f"Failed to update span with LLM metrics: {e}")
        
        logger.info(f"LLM response: {response_text}")
        return response_text
                
    except Exception as e:
        logger.error(f"LLM call error: {e}")
        if span:
            try:
                span.update(exception=e)
                span.update(output={
                    "status": "failed",
                    "error": str(e),
                    "model": AZURE_DEPLOYMENT_NAME,
                    "provider": "azure_openai"
                })
            except Exception:
                pass
        return "[ERROR: LLM call failed]"