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
        logger.info(f"LLM response: {resp['choices'][0]['message']['content']}")
        return True
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return False

def generate_llm_response(prompt: str, span=None):
    """
    Generator that yields tokens from Azure OpenAI streaming API via litellm.
    """
    try:
        # Enable streaming by setting stream=True
        response_stream = litellm.completion(
            model=f"azure/{AZURE_DEPLOYMENT_NAME}",
            messages=[{"role": "user", "content": prompt}],
            api_key=AZURE_API_KEY,
            api_base=AZURE_API_BASE,
            api_version=AZURE_API_VERSION,
            stream=True  # Enable streaming
        )
        for chunk in response_stream:
            # Extract content from chunk (adapt based on litellm's streaming format)
            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
            if content:
                yield content
    except Exception as e:
        logger.error(f"LLM streaming error: {e}")
        if span:
            try:
                span.update(metadata={"llm_error": str(e)})
            except Exception:
                pass
        yield "[ERROR: LLM streaming failed]"