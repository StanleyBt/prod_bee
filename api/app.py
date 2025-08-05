import logging
from typing import Optional, List
from fastapi import FastAPI, Query

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
from core.config import DEFAULT_MODULE
from api.lifespan import lifespan
from tracing import get_langfuse_client
from utils.welcome_questions import (
    load_welcome_questions,
    get_available_modules,
    get_available_roles,
    get_random_sample_questions
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


logger = logging.getLogger(__name__)

app = FastAPI(lifespan=lifespan, title="Multi-Tenant RAG API")

# --- MODELS ---
class QueryRequest(BaseModel):
    input: str
    tenant_id: str
    session_id: str
    module: Optional[str] = None
    role: str  # Make role required
    model_config = ConfigDict(extra="forbid")

# --- GRAPH NODES ---
def create_rag_flow():
    graph = Graph()
    langfuse = get_langfuse_client()

    def memory_retrieval_node(state: dict) -> dict:
        tenant_id = state.get("tenant_id", "")
        session_id = state.get("session_id", "")
        module = state.get("module") or DEFAULT_MODULE
        role = state.get("role", "")

        if langfuse:
            with langfuse.start_as_current_span(name="MemoryRetrieval") as span:
                span.update(input={
                    "tenant_id": tenant_id,
                    "session_id": session_id,
                    "module": module,
                    "role": role
                })
                try:
                    # Retrieve conversation memories
                    memories = retrieve_memories(
                        tenant_id=tenant_id,
                        session_id=session_id,
                        module=module,
                        role=role,
                        max_memories=10  # Limit to 10 most recent memories
                    )
                    state["memories"] = memories
                    
                    # Retrieve user preferences using Mem0
                    try:
                        user_context = get_user_context(
                            tenant_id=tenant_id,
                            user_id=session_id,  # Use session_id as user_id
                            session_id=session_id,
                            module=module,
                            role=role
                        )
                        state["user_preferences"] = user_context
                    except Exception as e:
                        logger.error(f"User preference retrieval failed: {e}")
                        state["user_preferences"] = ""
                    
                    span.update(output={
                        "memory_count": len(memories),
                        "memories": memories[:3] if memories else [],  # First 3 for context
                        "user_preferences": state.get("user_preferences", ""),
                        "session_id": session_id,
                        "module": module,
                        "role": role
                    })
                    logger.info(f"Retrieved {len(memories)} memories and user preferences for session: {session_id}")
                except Exception as e:
                    span.update(exception=e)
                    state["memories"] = []
                    state["user_preferences"] = ""
                    logger.error(f"Memory retrieval failed: {e}")
                return state
        else:
                        # Retrieve conversation memories
            memories = retrieve_memories(
                tenant_id=tenant_id,
                session_id=session_id,
                module=module,
                role=role,
                max_memories=10  # Limit to 10 most recent memories
            )
            state["memories"] = memories
            
            # Retrieve user preferences using Mem0
            try:
                user_context = get_user_context(
                    tenant_id=tenant_id,
                    user_id=session_id,  # Use session_id as user_id
                    session_id=session_id,
                    module=module,
                    role=role
                )
                state["user_preferences"] = user_context
            except Exception as e:
                logger.error(f"User preference retrieval failed: {e}")
                state["user_preferences"] = ""
            
            logger.info(f"Retrieved {len(memories)} memories and user preferences for session: {session_id}")
            return state

    def retriever_node(state: dict) -> dict:
        tenant_id = state.get("tenant_id", "")
        module = state.get("module") or DEFAULT_MODULE
        role = state.get("role", "")
        user_input = state["input"]

        def preprocess_context(chunks: List[str], max_chars: int = 1000) -> str:
            combined = " ".join(chunks)
            if len(combined) > max_chars:
                combined = combined[:max_chars] + "..."
            return combined

        if langfuse:
            with langfuse.start_as_current_span(name="VectorRetrieval") as span:
                span.update(input={
                    "query": user_input,
                    "tenant_id": tenant_id,
                    "module": module,
                    "role": role,
                    "top_k": 3
                })
                try:
                    result = retrieve_document_chunks(
                        tenant_id=tenant_id,
                        query=user_input,
                        module=module,
                        role=role,
                        top_k=3
                    )
                    

                    
                    context_summary = preprocess_context(result)
                    state["context"] = context_summary if context_summary else "No context found"
                    
                    # Enhanced output with detailed retrieval info
                    span.update(output={
                        "result_count": len(result) if result else 0,
                        "context_length": len(context_summary),
                        "context_preview": context_summary[:200] + "..." if len(context_summary) > 200 else context_summary,
                        "tenant_id": tenant_id,
                        "module": module,
                        "role": role,
                        "query_embedding_generated": True
                    })
                except Exception as e:
                    span.update(exception=e)
                    state["context"] = "Retrieval error"
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
                    query=user_input,
                    module=module,
                    role=role,
                    top_k=3
                )
                

                
                context_summary = preprocess_context(result)
                state["context"] = context_summary if context_summary else "No context found"
            except Exception:
                state["context"] = "Retrieval error"
            return state

    def llm_node(state: dict) -> dict:
        prompt = build_prompt(state, state["input"])
        if langfuse:
            with langfuse.start_as_current_span(name="LLMGeneration") as span:
                span.update(input={
                    "prompt_length": len(prompt),
                    "prompt_preview": prompt[:500] + "..." if len(prompt) > 500 else prompt,
                    "original_query": state["input"],
                    "tenant_id": state["tenant_id"],
                    "session_id": state["session_id"],
                    "module": state["module"],
                    "memory_count": len(state.get("memories", [])),
                    "context_length": len(state.get("context", ""))
                })
                try:
                    full_response = generate_llm_response(prompt, span)
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
def build_prompt(state: dict, user_input: str) -> str:
    memories = state.get("memories", [])
    user_preferences = state.get("user_preferences", "")
    
    # Debug logging to see what's being included
    logger.info(f"Building prompt with {len(memories)} memories and user preferences: {user_preferences[:100] if user_preferences else 'None'}")
    if memories:
        logger.info(f"Memory structure: {memories[0] if memories else 'No memories'}")
    
    # Handle memory format - use only the most recent memories
    memory_str = ""
    
    if memories:
        memory_entries = []
        # Use only the 3 most recent memories to keep context focused
        # Memories are already in descending order (newest first), so take first 3
        recent_memories = memories[:3] if len(memories) > 3 else memories
        
        # No hardcoded step detection - let the LLM handle conversation flow naturally
        
        for memory in recent_memories:
            if isinstance(memory, dict) and 'user' in memory and 'bot' in memory:
                user_msg = str(memory.get('user', ''))
                bot_msg = str(memory.get('bot', ''))
                memory_entries.append(f"Previous conversation:\nUser: {user_msg}\nAssistant: {bot_msg}")
        
        memory_str = "\n".join(memory_entries)
    
    if memory_str:
        logger.info(f"Conversation history preview: {memory_str[:200]}...")
    else:
        logger.info("No conversation history found")

    # --- PROMPT-ONLY RAG SAFETY: If context is empty, set a clear flag ---
    context = state.get('context', '')
    if not context or context.strip().lower() in ["no context found", "retrieval error"]:
        context = "No relevant information found for this module."
        # If module context is empty, do not include conversation history
        memory_str = ""

    # Determine the appropriate ending based on context and conversation state
    ending_instruction = ""
    
    # Simple, natural conversation flow
    has_conversation_history = bool(memory_str.strip())
    
    if has_conversation_history:
        ending_instruction = "Continue the conversation naturally, building on our previous discussion and providing helpful, context-aware responses."
    else:
        ending_instruction = "Engage with the user's input in a helpful and professional manner."
    
    prompt = (
        "You are a smart, friendly, and professional AI assistant built for a modern SaaS platform. "
        "Instructions:\n"
        "- Understand the user's intent using context and conversation history.\n"
        "- Respond clearly and concisely using markdown formatting — include bullet points, numbered steps, or headings if useful.\n"
        "- Avoid overwhelming users — provide just enough detail to address their current need.\n"
        "- Adapt your tone to be professional yet approachable.\n"
        "- Respect the user's role (e.g., employee, manager, vendor, HR) to customize instructions.\n"
        "- Keep responses relevant — don't repeat earlier answers unless needed for clarity.\n"
        "- Your responses should be modular and context-aware.\n"
        "- IMPORTANT: If the module context does not contain relevant information, do NOT attempt to answer the user's question. Instead, politely inform the user that you cannot answer and ask them to rephrase or ask about the selected module.\n"
        "- Focus your responses strictly on the user's selected module and avoid discussing other modules.\n"
        "- IMPORTANT: You have access to conversation history below. Use it to provide context-aware responses.\n"
        "- Be conversational and engaging, but stay focused on the user's needs.\n"
        "- IMPORTANT: Respond naturally to user inputs, understanding context from conversation history.\n"
        "- IMPORTANT: Adapt your response style based on the user's preferences below.\n\n"
        f"User Role (if known): {state.get('role', 'unknown')}\n"
        f"User Preferences: {user_preferences if user_preferences else 'No specific preferences found'}\n"
        f"Module Context (if applicable): {context}\n"

    )
    if memory_str:
        prompt += f"Conversation History:\n{memory_str}\n\n"
    prompt += (
        f"User Input:\n{user_input}\n\n"
        f"{ending_instruction}"
    )
    return prompt

# --- FASTAPI ENDPOINTS ---
@app.post("/query")
async def query_rag(request: QueryRequest):
    state = {
        "input": request.input,
        "tenant_id": request.tenant_id,
        "session_id": request.session_id,
        "module": request.module or DEFAULT_MODULE,
        "role": request.role # Pass role to the state
    }
    state = rag_flow_executor.invoke(state)
    full_response = state.get("llm_response", "[ERROR: No LLM response]")
    
    # Store memory in both systems
    # 1. Store in per-tenant storage (existing system)
    store_memory(
        tenant_id=state["tenant_id"],
        user_input=request.input,
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
            user_input=request.input,
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
    
    return {"response": full_response}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/welcome")
async def welcome(tenant_id: str = Query(..., description="Tenant ID")):
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
class ClearConversationsRequest(BaseModel):
    tenant_id: str
    session_id: str
    model_config = ConfigDict(extra="forbid")



@app.post("/clear-conversations")
async def clear_conversations(request: ClearConversationsRequest):
    """
    Clear all Weaviate conversation memories for a specific session.
    Preserves Mem0 memories - only cleans up Weaviate conversation history.
    """
    try:
        result = clear_session_conversations(
            tenant_id=request.tenant_id,
            session_id=request.session_id
        )
        
        if result["success"]:
            return {
                "success": True,
                "message": result["message"],
                "deleted_count": result["deleted_count"],
                "tenant_id": request.tenant_id,
                "session_id": request.session_id
            }
        else:
            return {
                "success": False,
                "error": result["error"],
                "tenant_id": request.tenant_id,
                "session_id": request.session_id
            }
            
    except Exception as e:
        logger.error(f"Error in clear_conversations endpoint: {e}")
        return {
            "success": False,
            "error": f"Internal server error: {str(e)}",
            "tenant_id": request.tenant_id,
            "session_id": request.session_id
        }


