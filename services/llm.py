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
            messages=[{"role": "user", "content": "Test connection"}],
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

def generate_llm_response(prompt: str, span=None, session_id=None):
    """Generate a response from Azure OpenAI LLM via litellm with cost tracking."""
    from tracing import get_langfuse_client
    
    langfuse = get_langfuse_client()
    
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
        
        # Extract usage information for cost calculation
        usage_info = {}
        if hasattr(resp, 'usage') and resp.usage:
            usage_info = {
                "input": getattr(resp.usage, 'prompt_tokens', 0),
                "output": getattr(resp.usage, 'completion_tokens', 0),
                "total": getattr(resp.usage, 'total_tokens', 0)
            }
        
        # Calculate cost for your GPT-4.1 Mini model
        cost_info = {}
        if usage_info:
            cost_info = calculate_azure_openai_cost(
                model=AZURE_DEPLOYMENT_NAME,
                input_tokens=usage_info["input"],
                output_tokens=usage_info["output"]
            )
        
        # Create Langfuse generation for Model Costs dashboard using correct API
        if langfuse and usage_info:
            try:
                with langfuse.start_as_current_generation(
                    name="azure-openai-completion",
                    model=AZURE_DEPLOYMENT_NAME,
                    input=prompt[:500] + "..." if len(prompt) > 500 else prompt
                ) as generation:
                    # Add user_id to the current trace for user tracking
                    langfuse.update_current_trace(user_id=session_id or "session_user")
                    generation.update(
                        output=response_text[:500] + "..." if len(response_text) > 500 else response_text,
                        usage_details={
                            "input": usage_info["input"],
                            "output": usage_info["output"],
                            "total": usage_info["total"]
                        },
                        cost_details={
                            "input": cost_info["input_cost"],
                            "output": cost_info["output_cost"],
                            "total": cost_info["total_cost"]
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to create generation for cost tracking: {e}")
        
        # Update the existing LLMGeneration span with cost information
        if span:
            try:
                span.update(output={
                    "model": AZURE_DEPLOYMENT_NAME,
                    "response_length": len(response_text) if response_text else 0,
                    "provider": "azure_openai",
                    "usage": usage_info,
                    "cost": cost_info,
                    "response_preview": response_text[:300] + "..." if len(response_text) > 300 else response_text
                })
                # Also update metadata with cost information
                span.update(metadata={
                    "model": AZURE_DEPLOYMENT_NAME,
                    "provider": "azure_openai",
                    "usage": usage_info,
                    "cost": cost_info,
                    "span_type": "llm_generation_with_cost"
                })
            except Exception as e:
                logger.warning(f"Failed to update span with LLM metrics: {e}")
        
        logger.info(f"LLM response: {response_text}")
        # Remove cost logging - costs will be visible in Langfuse dashboard
        
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


def calculate_azure_openai_cost(model: str, input_tokens: int, output_tokens: int) -> dict:
    """
    Calculate cost based on Azure OpenAI pricing.
    Pricing as of 2024 (per 1K tokens):
    - GPT-4: $0.03 input, $0.06 output
    - GPT-4.1 Mini: $0.00015 input, $0.0006 output (your model)
    - GPT-4 Turbo: $0.01 input, $0.03 output
    - GPT-3.5 Turbo: $0.0015 input, $0.002 output
    """
    # Azure OpenAI pricing per 1K tokens (as of 2024)
    pricing = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4.1-mini": {"input": 0.00015, "output": 0.0006},  # Your model
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-35-turbo": {"input": 0.0015, "output": 0.002},
        "gpt-35-turbo-16k": {"input": 0.003, "output": 0.004},
    }
    
    # Normalize model name for lookup
    model_key = model.lower().replace("-", "").replace("_", "")
    
    # Find matching pricing (handle variations in model names)
    model_pricing = None
    for key, price in pricing.items():
        if key in model_key or model_key in key:
            model_pricing = price
            break
    
    # Default to GPT-4.1 Mini pricing if model not found (since that's your model)
    if not model_pricing:
        logger.warning(f"Unknown model {model}, using GPT-4.1 Mini pricing")
        model_pricing = pricing["gpt-4.1-mini"]
    
    # Calculate costs
    input_cost = (input_tokens / 1000) * model_pricing["input"]
    output_cost = (output_tokens / 1000) * model_pricing["output"]
    total_cost = input_cost + output_cost
    
    return {
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(total_cost, 6),
        "model": model,
        "pricing_tier": model_pricing
    }