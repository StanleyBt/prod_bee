import logging
from typing import Optional, List, Dict
from fastapi import FastAPI, Query, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from core.logging_config import initialize_logging, get_logger
from core.error_handling import (
    RAGException, ValidationError, RateLimitError, ServiceUnavailableError,
    TenantNotFoundError, rag_exception_handler, validation_exception_handler,
    http_exception_handler, general_exception_handler
)
from core.middleware import RequestCorrelationMiddleware, MetricsMiddleware

from pydantic import BaseModel, ConfigDict
from langgraph.graph import Graph, END
from services.per_tenant_storage import (
    retrieve_document_chunks,
    store_memory,
    retrieve_memories,
    initialize_per_tenant_storage,
    clear_session_conversations
)

from services.mem0_service import (
    get_user_context,
    store_user_memory
)
from services.llm import generate_llm_response
from core.config import DEFAULT_MODULE, MAX_CONTEXT_CHARS, MAX_CHUNKS_PER_QUERY
from api.lifespan import lifespan
from tracing import get_langfuse_client
from utils.welcome_questions import (
    load_welcome_questions,
    get_available_modules,
    get_available_roles,
    get_random_sample_questions
)

# Import validation and rate limiting modules
from core.validation import (
    SanitizedQueryRequest,
    SanitizedClearRequest,
    sanitize_text
)
from core.rate_limiting import (
    check_user_rate_limit,
    RateLimitConfig
)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Suppress SWIG-related deprecation warnings
warnings.filterwarnings("ignore", message=".*builtin type.*has no __module__ attribute")
# Suppress socket resource warnings (handled by proper cleanup)
warnings.filterwarnings("ignore", category=ResourceWarning, message=".*unclosed.*socket.*")
# Suppress Weaviate connection warnings (handled by proper cleanup)
warnings.filterwarnings("ignore", message=".*The connection to Weaviate was not closed properly.*")
warnings.filterwarnings("ignore", message=".*Con004.*")
warnings.filterwarnings("ignore", message=".*connection.*was not closed properly.*")


# Initialize logging
initialize_logging()
logger = get_logger(__name__)

app = FastAPI(lifespan=lifespan, title="Multi-Tenant RAG API")

# Add middleware
app.add_middleware(RequestCorrelationMiddleware)
metrics_middleware = MetricsMiddleware(app)
app.add_middleware(MetricsMiddleware)

# Add exception handlers
app.add_exception_handler(RAGException, rag_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Configure rate limiting
rate_limit_config = RateLimitConfig(
    requests_per_minute=60,
    requests_per_hour=1000,
    requests_per_day=10000,
    burst_limit=10
)

# Create rate limiter for endpoint-level rate limiting
from core.rate_limiting import RateLimiter
rate_limiter = RateLimiter(rate_limit_config)

# --- MODELS ---
# Keep original models for backward compatibility, but use sanitized versions in endpoints
class QueryRequest(BaseModel):
    input: str
    tenant_id: str
    session_id: str
    module: Optional[str] = None
    role: str  # Make role required
    model_config = ConfigDict(extra="forbid")

class ClearConversationsRequest(BaseModel):
    tenant_id: str
    session_id: str
    model_config = ConfigDict(extra="forbid")

# --- RATE LIMITING DEPENDENCIES ---
def check_rate_limit(request: Request, tenant_id: str, session_id: str):
    """Check rate limit for the current request"""
    endpoint = request.url.path
    return check_user_rate_limit(tenant_id, session_id, endpoint)

# --- VALIDATION DEPENDENCIES ---
async def validate_query_request(request: QueryRequest) -> SanitizedQueryRequest:
    """Validate and sanitize query request"""
    try:
        return SanitizedQueryRequest(**request.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

async def validate_clear_request(request: ClearConversationsRequest) -> SanitizedClearRequest:
    """Validate and sanitize clear request"""
    try:
        return SanitizedClearRequest(**request.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- GRAPH NODES ---
def create_rag_flow():
    graph = Graph()
    langfuse = get_langfuse_client()

    def _retrieve_memories_and_preferences(tenant_id: str, session_id: str, module: str, role: str) -> tuple[list, str]:
        """
        Retrieve conversation memories and user preferences.
        Returns (memories, user_preferences) tuple.
        """
        # Retrieve conversation memories
        memories = retrieve_memories(
            tenant_id=tenant_id,
            session_id=session_id,
            module=module,
            role=role,
            max_memories=10  # Limit to 10 most recent memories
        )
        
        # Retrieve user preferences using Mem0
        user_preferences = ""
        try:
            user_context = get_user_context(
                tenant_id=tenant_id,
                user_id=session_id,  # Use session_id as user_id
                session_id=session_id,
                module=module,
                role=role
            )
            user_preferences = user_context
        except Exception as e:
            logger.error(f"User preference retrieval failed: {e}")
            user_preferences = ""
        
        logger.info(f"Retrieved {len(memories)} memories and user preferences for session: {session_id}")
        return memories, user_preferences

    def memory_retrieval_node(state: dict) -> dict:
        tenant_id = state.get("tenant_id", "")
        session_id = state.get("session_id", "")
        module = state.get("module") or DEFAULT_MODULE
        role = state.get("role", "")

        if langfuse:
            with langfuse.start_as_current_span(
                name="MemoryRetrieval",
                metadata={
                    "component": "memory_system",
                    "operation": "retrieve_user_memories_and_preferences",
                    "tenant_id": tenant_id,
                    "session_id": session_id,
                    "module": module,
                    "role": role
                }
            ) as span:
                # Add user_id to the current trace for user tracking
                langfuse.update_current_trace(user_id=session_id)
                span.update(input={
                    "tenant_id": tenant_id,
                    "session_id": session_id,
                    "module": module,
                    "role": role,
                    "operation": "retrieve_memories_and_preferences"
                })
                try:
                    memories, user_preferences = _retrieve_memories_and_preferences(
                        tenant_id, session_id, module, role
                    )
                    state["memories"] = memories
                    state["user_preferences"] = user_preferences
                    
                    span.update(output={
                        "memory_count": len(memories),
                        "memories": memories[:3] if memories else [],  # First 3 for context
                        "user_preferences": user_preferences,
                        "total_memory_length": sum(len(mem) for mem in memories),
                        "status": "success"
                    })
                    span.update(metadata={
                        "memory_retrieval_success": True,
                        "memory_count": len(memories),
                        "has_user_preferences": bool(user_preferences)
                    })
                except Exception as e:
                    span.update(exception=e)
                    state["memories"] = []
                    state["user_preferences"] = ""
                    logger.error(f"Memory retrieval failed: {e}")
                return state
        else:
            try:
                memories, user_preferences = _retrieve_memories_and_preferences(
                    tenant_id, session_id, module, role
                )
                state["memories"] = memories
                state["user_preferences"] = user_preferences
            except Exception as e:
                state["memories"] = []
                state["user_preferences"] = ""
                logger.error(f"Memory retrieval failed: {e}")
            return state

    def retriever_node(state: dict) -> dict:
        tenant_id = state.get("tenant_id", "")
        session_id = state.get("session_id", "")
        module = state.get("module") or DEFAULT_MODULE
        role = state.get("role", "")
        user_input = state["input"]
        memories = state.get("memories", [])
        
        # Build contextual query using conversation history
        def build_contextual_query(current_input: str, memories: List[Dict]) -> str:
            """
            Build an enhanced query that includes conversation context
            to improve document retrieval accuracy.
            """
            if not memories:
                return current_input
            
            # Skip enhancement for very short queries (likely greetings or simple responses)
            if len(current_input.split()) <= 2:
                logger.info(f"Skipping enhancement for short query: '{current_input}'")
                return current_input
            
            # Extract key terms from the most recent conversation
            recent_memory = memories[0] if memories else {}
            user_question = recent_memory.get('user', '')
            bot_response = recent_memory.get('bot', '')
            
            # Extract meaningful terms from the conversation (non-generic words)
            import re
            
            # Combine user question and bot response for analysis
            conversation_text = f"{user_question} {bot_response}".lower()
            
            # Basic stop words (minimal set)
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'how', 'when', 'where', 'why', 'who', 'which', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
            
            # Extract words that are likely to be important context
            words = re.findall(r'\b[a-zA-Z]{3,}\b', conversation_text)
            meaningful_terms = [word for word in words if word not in stop_words and len(word) > 2]
            
            # Take the most relevant terms (limit to avoid too long queries)
            key_terms = list(dict.fromkeys(meaningful_terms))[:3]  # Limit to 3 terms for better focus
            
            # Only enhance if we have meaningful terms
            if key_terms and len(key_terms) >= 1:
                contextual_query = f"{' '.join(key_terms)} {current_input}"
                logger.info(f"Enhanced query: '{current_input}' -> '{contextual_query}'")
                return contextual_query
            
            logger.info(f"No meaningful context found for: '{current_input}'")
            return current_input
        
        # Use enhanced query for better document retrieval
        enhanced_query = build_contextual_query(user_input, memories)

        def _get_video_segment_from_mapping(video_mapping: dict, document_filename: str, user_query: str) -> Optional[dict]:
            """
            Get video segment from pre-loaded mapping without file I/O.
            
            Args:
                video_mapping: Pre-loaded video mapping dictionary
                document_filename: The filename of the document
                user_query: The user's query to match against segments
                
            Returns:
                Dictionary with video_url, start time, and topic if found, None otherwise
            """
            video_info = video_mapping.get(document_filename)
            
            if not video_info:
                return None
            
            # Handle old format (simple string)
            if isinstance(video_info, str):
                return {
                    "url": video_info,
                    "start": 0,
                    "topic": "full_video"
                }
            
            # Handle new format with segments
            if isinstance(video_info, dict) and "video_url" in video_info:
                video_url = video_info["video_url"]
                segments = video_info.get("segments", {})
                
                if not segments:
                    return {
                        "url": video_url,
                        "start": 0,
                        "topic": "full_video"
                    }
                
                # Find best matching segment
                from utils.video_mapping import find_best_segment
                best_segment = find_best_segment(user_query, segments)
                if best_segment:
                    return {
                        "url": video_url,
                        "start": best_segment["start"],
                        "topic": best_segment["topic"]
                    }
                else:
                    return {
                        "url": video_url,
                        "start": 0,
                        "topic": "full_video"
                    }
            
            return None

        def preprocess_context(chunks: List[str], max_chars: int = MAX_CONTEXT_CHARS) -> tuple[str, List[dict]]:
            if not chunks:
                return "", []
            
            # Load video mapping once for this tenant (cached)
            from utils.video_mapping import load_video_mapping, find_best_segment
            video_mapping = load_video_mapping(tenant_id)
            
            # Extract document filenames from chunks and get their video segments
            video_segments = []
            for chunk in chunks:
                # Extract filename from chunk context
                if "File:" in chunk:
                    try:
                        file_start = chunk.find("File:") + 5
                        file_end = chunk.find(" |", file_start)
                        if file_end == -1:
                            file_end = chunk.find("]", file_start)
                        if file_end == -1:
                            file_end = len(chunk)
                        filename = chunk[file_start:file_end].strip()
                        
                        # Get video segment from cached mapping
                        video_segment = _get_video_segment_from_mapping(video_mapping, filename, user_input)
                        if video_segment:
                            # Add document info to segment
                            video_segment["document"] = filename
                            # Avoid duplicates
                            if not any(seg["url"] == video_segment["url"] and seg["start"] == video_segment["start"] for seg in video_segments):
                                video_segments.append(video_segment)
                    except Exception as e:
                        logger.debug(f"Failed to extract filename from chunk: {e}")
            
            # Intelligent chunk combination with better token management
            # Prioritize complete chunks over truncation
            combined_chunks = []
            current_length = 0
            
            for chunk in chunks:
                chunk_length = len(chunk)
                # If adding this chunk would exceed limit, stop
                if current_length + chunk_length > max_chars:
                    break
                combined_chunks.append(chunk)
                current_length += chunk_length
            
            if combined_chunks:
                return " ".join(combined_chunks), video_segments
            else:
                # If even the first chunk is too long, truncate it
                return chunks[0][:max_chars] + "..." if chunks else "", video_segments

        if langfuse:
            with langfuse.start_as_current_span(
                name="VectorRetrieval",
                metadata={
                    "component": "vector_search",
                    "operation": "retrieve_document_chunks",
                    "tenant_id": tenant_id,
                    "module": module,
                    "role": role
                }
            ) as span:
                # Add user_id to the current trace for user tracking
                langfuse.update_current_trace(user_id=session_id)
                span.update(input={
                    "query": user_input,
                    "enhanced_query": enhanced_query,
                    "tenant_id": tenant_id,
                    "module": module,
                    "role": role,
                    "top_k": MAX_CHUNKS_PER_QUERY,
                    "query_length": len(user_input),
                    "enhanced_query_length": len(enhanced_query),
                    "operation": "vector_search"
                })
                try:
                    result = retrieve_document_chunks(
                        tenant_id=tenant_id,
                        query=enhanced_query,
                        module=module,
                        role=role,
                        top_k=MAX_CHUNKS_PER_QUERY,
                        session_id=session_id
                    )
                    
                    context_summary, video_segments = preprocess_context(result)
                    state["context"] = context_summary if context_summary else "No context found"
                    state["video_segments"] = video_segments  # NEW: Store video segments in state
                    
                    # Log retrieval summary
                    if enhanced_query != user_input:
                        logger.info(f"Retrieved {len(result) if result else 0} chunks for enhanced query '{enhanced_query}' (original: '{user_input}'): {len(context_summary)} chars, {len(video_segments)} video segments")
                    else:
                        logger.info(f"Retrieved {len(result) if result else 0} chunks for '{user_input}': {len(context_summary)} chars, {len(video_segments)} video segments")
                    
                    # Enhanced output with detailed retrieval info
                    span.update(output={
                        "result_count": len(result) if result else 0,
                        "context_length": len(context_summary),
                        "context_preview": context_summary[:200] + "..." if len(context_summary) > 200 else context_summary,
                        "video_segments_count": len(video_segments),
                        "tenant_id": tenant_id,
                        "module": module,
                        "role": role,
                        "query_embedding_generated": True
                    })
                except Exception as e:
                    span.update(exception=e)
                    state["context"] = "Retrieval error"
                    state["video_segments"] = []  # NEW: Empty video segments on error
                    span.update(output={
                        "result_count": 0,
                        "error": str(e),
                        "tenant_id": tenant_id,
                        "module": module,
                        "role": role
                    })
                return state
        else:
            try:
                result = retrieve_document_chunks(
                    tenant_id=tenant_id,
                    query=enhanced_query,
                    module=module,
                    role=role,
                    top_k=MAX_CHUNKS_PER_QUERY,
                    session_id=session_id
                )
                
                context_summary, video_segments = preprocess_context(result)
                state["context"] = context_summary if context_summary else "No context found"
                state["video_segments"] = video_segments  # NEW: Store video segments in state
                
                # Log retrieval summary
                if enhanced_query != user_input:
                    logger.info(f"Retrieved {len(result) if result else 0} chunks for enhanced query '{enhanced_query}' (original: '{user_input}'): {len(context_summary)} chars, {len(video_segments)} video segments")
                else:
                    logger.info(f"Retrieved {len(result) if result else 0} chunks for '{user_input}': {len(context_summary)} chars, {len(video_segments)} video segments")
            except Exception:
                state["context"] = "Retrieval error"
                state["video_segments"] = []  # NEW: Empty video segments on error
            return state

    def llm_node(state: dict) -> dict:
        prompt = build_prompt(state, state["input"])
        if langfuse:
            with langfuse.start_as_current_span(
                name="LLMGeneration",
                metadata={
                    "component": "llm_service",
                    "operation": "generate_response",
                    "tenant_id": state["tenant_id"],
                    "session_id": state["session_id"],
                    "module": state["module"],
                    "role": state["role"]
                }
            ) as span:
                # Add user_id to the current trace for user tracking
                langfuse.update_current_trace(user_id=state["session_id"])
                span.update(input={
                    "prompt_length": len(prompt),
                    "prompt_preview": prompt[:500] + "..." if len(prompt) > 500 else prompt,
                    "original_query": state["input"],
                    "tenant_id": state["tenant_id"],
                    "session_id": state["session_id"],
                    "module": state["module"],
                    "role": state["role"],
                    "memory_count": len(state.get("memories", [])),
                    "context_length": len(state.get("context", "")),
                    "user_preferences_length": len(state.get("user_preferences", "")),
                    "operation": "llm_generation"
                })
                try:
                    full_response = generate_llm_response(prompt, span, state["session_id"])
                    if full_response:
                        span.update(output={
                            "status": "completed",
                            "response_length": len(full_response),
                            "response_preview": full_response[:300] + "..." if len(full_response) > 300 else full_response,
                            "tenant_id": state["tenant_id"],
                            "session_id": state["session_id"],
                            "module": state["module"],
                            "role": state["role"]
                        })
                        state["llm_response"] = full_response
                        
                    else:
                        state["llm_response"] = "[ERROR: Empty response from LLM]"
                except Exception as e:
                    span.record_exception(e)
                    state["llm_response"] = "[ERROR: Response generation failed]"
        else:
            try:
                full_response = generate_llm_response(prompt)
                state["llm_response"] = full_response
                
            except Exception:
                state["llm_response"] = "[ERROR: Response generation failed]"
        return state

    graph.add_node("memory_retrieval", memory_retrieval_node)
    graph.add_node("retrieve", retriever_node)
    graph.add_node("llm", llm_node)

    graph.set_entry_point("memory_retrieval")
    graph.add_edge("memory_retrieval", "retrieve")
    graph.add_edge("retrieve", "llm")
    graph.add_edge("llm", END)
    return graph.compile()

rag_flow_executor = create_rag_flow()



# --- PROMPT BUILDER ---
from utils.prompt_builder import build_prompt

# --- FASTAPI ENDPOINTS ---
@app.post("/query")
async def query_rag(
    request: QueryRequest,
    validated_request: SanitizedQueryRequest = Depends(validate_query_request),
    fastapi_request: Request = None
):
    # Get Langfuse client for top-level tracing
    langfuse = get_langfuse_client()
    
    # Check rate limit after validation
    if fastapi_request:
        check_rate_limit(fastapi_request, validated_request.tenant_id, validated_request.session_id)
    
    # Use validated and sanitized data
    state = {
        "input": validated_request.input,
        "tenant_id": validated_request.tenant_id,
        "session_id": validated_request.session_id,
        "module": validated_request.module or DEFAULT_MODULE,
        "role": validated_request.role
    }
    
    # Execute the RAG flow
    state = rag_flow_executor.invoke(state)
    full_response = state.get("llm_response", "[ERROR: No LLM response]")
    
    # Store memory in both systems
    # 1. Store in per-tenant storage (existing system)
    store_memory(
        tenant_id=state["tenant_id"],
        user_input=validated_request.input,
        response_text=full_response,
        session_id=state["session_id"],
        module=state["module"],
        role=state["role"]
    )
    
    # 2. Store in Mem0 for user preference context
    try:
        mem0_success = store_user_memory(
            tenant_id=state["tenant_id"],
            user_id=state["session_id"],  # Use session_id as user_id
            user_input=validated_request.input,
            response_text=full_response,
            session_id=state["session_id"],
            module=state["module"],
            role=state["role"]
        )
        if mem0_success:
            logger.info("Successfully stored user memory in Mem0")
        else:
            logger.warning("Failed to store user memory in Mem0")
    except Exception as e:
        logger.error(f"Exception storing user memory in Mem0: {e}")
    
    # Get video segments from state
    video_segments = state.get("video_segments", [])
    
    # Return response with video segments
    return {
        "response": full_response,
        "video_segments": video_segments  # NEW: Include video segments in response
    }


@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/metrics")
async def get_metrics():
    """Get application metrics."""
    return metrics_middleware.get_metrics()


@app.get("/cache/stats")
async def get_cache_stats():
    """Get embedding cache statistics."""
    try:
        from utils.embeddings import get_embedding_cache_stats
        stats = get_embedding_cache_stats()
        if stats:
            return {
                "status": "success",
                "cache_stats": stats
            }
        else:
            return {
                "status": "error",
                "message": "Cache statistics not available"
            }
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        return {
            "status": "error",
            "message": f"Failed to get cache statistics: {str(e)}"
        }

@app.post("/cache/clear")
async def clear_cache():
    """Clear the embedding cache."""
    try:
        from utils.embeddings import clear_embedding_cache
        clear_embedding_cache()
        return {
            "status": "success",
            "message": "Embedding cache cleared successfully"
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return {
            "status": "error",
            "message": f"Failed to clear cache: {str(e)}"
        }



@app.get("/welcome")
async def welcome(
    tenant_id: str = Query(..., description="Tenant ID"),
    request: Request = None
):
    # Validate and sanitize tenant_id
    try:
        from core.validation import sanitize_text
        sanitized_tenant_id = sanitize_text(tenant_id, max_length=50)
        if not sanitized_tenant_id:
            raise HTTPException(status_code=400, detail="Invalid tenant ID")
        tenant_id = sanitized_tenant_id.upper()  # Normalize to uppercase
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid tenant ID: {str(e)}")
    
    # Check rate limit if request is available
    if request:
        check_rate_limit(request, tenant_id, "welcome")
    """
    Welcome endpoint for tenants with all sample questions.
    
    Returns all questions for the tenant - frontend will handle role-based filtering.
    
    Args:
        tenant_id: The tenant ID
        
    Returns:
        - Welcome message
        - Available modules and roles
        - All sample questions organized by module and role
        - Quick start questions
    """
    try:
        # Get available modules and roles
        available_modules = get_available_modules(tenant_id)
        available_roles = get_available_roles(tenant_id)
        
        # Get ALL questions (no role filtering - frontend will handle this)
        all_questions = load_welcome_questions(tenant_id)
        
        # Get random sample questions for quick start (from all roles)
        quick_start_questions = get_random_sample_questions(tenant_id, "hr", count=5)  # Use hr to get all questions
        
        # Build welcome message
        message = f"Welcome to {tenant_id}! I'm here to help you with any questions."
        
        return {
            "message": message,
            "tenant_id": tenant_id,
            "available_modules": available_modules,
            "available_roles": available_roles,
            "questions": all_questions,
            "quick_start": quick_start_questions,
            "total_questions": sum(
                len(role_qs) for mod_qs in all_questions.values() for role_qs in mod_qs.values()
            ),
            "note": "Frontend should filter questions based on user role"
        }
        
    except Exception as e:
        logger.error(f"Error in welcome endpoint for tenant {tenant_id}: {e}")
        return {
            "error": "Failed to load sample questions",
            "message": f"Welcome to {tenant_id}! I'm here to help you with any questions.",
            "tenant_id": tenant_id,
            "available_modules": [],
            "available_roles": [],
            "questions": {},
            "quick_start": []
        }

# --- CLEANUP ENDPOINTS ---
@app.post("/clear-conversations")
async def clear_conversations(
    request: ClearConversationsRequest,
    validated_request: SanitizedClearRequest = Depends(validate_clear_request),
    fastapi_request: Request = None
):
    # Check rate limit after validation
    if fastapi_request:
        check_rate_limit(fastapi_request, validated_request.tenant_id, validated_request.session_id)
    """
    Clear all Weaviate conversation memories for a specific session.
    Preserves Mem0 memories - only cleans up Weaviate conversation history.
    """
    try:
        result = clear_session_conversations(
            tenant_id=validated_request.tenant_id,
            session_id=validated_request.session_id
        )
        
        if result["success"]:
            return {
                "success": True,
                "message": result["message"],
                "deleted_count": result["deleted_count"],
                "tenant_id": validated_request.tenant_id,
                "session_id": validated_request.session_id
            }
        else:
            return {
                "success": False,
                "error": result["error"],
                "tenant_id": validated_request.tenant_id,
                "session_id": validated_request.session_id
            }
            
    except Exception as e:
        logger.error(f"Error in clear_conversations endpoint: {e}")
        return {
            "success": False,
            "error": f"Internal server error: {str(e)}",
            "tenant_id": validated_request.tenant_id,
            "session_id": validated_request.session_id
        }


