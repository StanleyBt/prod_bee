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

def generate_llm_response(user_input: str, span=None) -> str:
    """Generate a response from the Azure LLM."""
    try:
        resp = litellm.completion(
            model=f"azure/{AZURE_DEPLOYMENT_NAME}",
            messages=[{"role": "user", "content": user_input}],
            api_key=AZURE_API_KEY,
            api_base=AZURE_API_BASE,
            api_version=AZURE_API_VERSION,
        )
        response_text = resp["choices"][0]["message"]["content"]
        logger.info(f"LLM response: {response_text}")
        return response_text
    except Exception as e:
        logger.error(f"LLM call error: {e}")
        if span:
            try:
                span.update(metadata={"llm_error": str(e)})
            except Exception:
                pass
        return "[ERROR: LLM call failed]"
