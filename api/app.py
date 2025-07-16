import logging
from typing import Optional, List
from fastapi import FastAPI, Query

from pydantic import BaseModel, ConfigDict
from langgraph.graph import Graph, END
from services.per_tenant_storage import (
    retrieve_document_chunks,
    store_memory,
    retrieve_memories,
    initialize_per_tenant_storage
)
from services.llm import generate_llm_response
from core.config import DEFAULT_MODULE
from api.lifespan import lifespan
from tracing import get_langfuse_client

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


logger = logging.getLogger(__name__)

app = FastAPI(lifespan=lifespan, title="Multi-Tenant RAG API")

# --- MODELS ---
class QueryRequest(BaseModel):
    input: str
    tenant_id: str
    session_id: str
    module: Optional[str] = None
    model_config = ConfigDict(extra="forbid")

# --- GRAPH NODES ---
def create_rag_flow():
    graph = Graph()
    langfuse = get_langfuse_client()

    def memory_retrieval_node(state: dict) -> dict:
        tenant_id = state.get("tenant_id", "")
        session_id = state.get("session_id", "")
        module = state.get("module") or DEFAULT_MODULE

        if langfuse:
            with langfuse.start_as_current_span(name="MemoryRetrieval") as span:
                span.update(input={
                    "tenant_id": tenant_id,
                    "session_id": session_id,
                    "module": module
                })
                try:
                    memories = retrieve_memories(
                        tenant_id=tenant_id,
                        session_id=session_id,
                        module=module,
                        max_memories=5  # Limit to 5 most recent memories
                    )
                    state["memories"] = memories
                    span.update(output={
                        "memory_count": len(memories),
                        "memories": memories[:3] if memories else [],  # First 3 for context
                        "session_id": session_id,
                        "module": module
                    })
                    logger.info(f"Retrieved {len(memories)} memories for session: {session_id}")
                except Exception as e:
                    span.update(exception=e)
                    state["memories"] = []
                    logger.error(f"Memory retrieval failed: {e}")
                return state
        else:
            memories = retrieve_memories(
                tenant_id=tenant_id,
                session_id=session_id,
                module=module,
                max_memories=5  # Limit to 5 most recent memories
            )
            state["memories"] = memories
            logger.info(f"Retrieved {len(memories)} memories for session: {session_id}")
            return state

    def retriever_node(state: dict) -> dict:
        tenant_id = state.get("tenant_id", "")
        module = state.get("module") or DEFAULT_MODULE
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
                    "top_k": 3
                })
                try:
                    result = retrieve_document_chunks(
                        tenant_id=tenant_id,
                        query=user_input,
                        module=module,
                        top_k=3
                    )
                    
                    # --- DEBUG: Print retrieved chunks content ---
                    logger.info(f"🔍 DEBUG: Retrieved {len(result)} chunks for query: '{user_input}' in module: {module}")
                    for i, chunk in enumerate(result):
                        logger.info(f"🔍 DEBUG: Chunk {i+1}: {chunk[:200]}...")
                    # --- END DEBUG ---
                    
                    context_summary = preprocess_context(result)
                    state["context"] = context_summary if context_summary else "No context found"
                    
                    # Enhanced output with detailed retrieval info
                    span.update(output={
                        "result_count": len(result) if result else 0,
                        "context_length": len(context_summary),
                        "context_preview": context_summary[:200] + "..." if len(context_summary) > 200 else context_summary,
                        "tenant_id": tenant_id,
                        "module": module,
                        "query_embedding_generated": True
                    })
                except Exception as e:
                    span.update(exception=e)
                    state["context"] = "Retrieval error"
                    span.update(output={
                        "result_count": 0,
                        "error": str(e),
                        "tenant_id": tenant_id,
                        "module": module
                    })
                return state
        else:
            try:
                result = retrieve_document_chunks(
                    tenant_id=tenant_id,
                    query=user_input,
                    module=module,
                    top_k=3
                )
                
                # --- DEBUG: Print retrieved chunks content ---
                logger.info(f"🔍 DEBUG: Retrieved {len(result)} chunks for query: '{user_input}' in module: {module}")
                for i, chunk in enumerate(result):
                    logger.info(f"🔍 DEBUG: Chunk {i+1}: {chunk[:200]}...")
                # --- END DEBUG ---
                
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
                            "module": state["module"]
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
    
    # Debug logging to see what's being included
    logger.info(f"Building prompt with {len(memories)} memories")
    if memories:
        logger.info(f"Memory structure: {memories[0] if memories else 'No memories'}")
    
    # Handle memory format - use only the most recent memories
    memory_str = ""
    current_step = None
    
    if memories:
        memory_entries = []
        # Use only the 3 most recent memories to keep context focused
        # Memories are already in descending order (newest first), so take first 3
        recent_memories = memories[:3] if len(memories) > 3 else memories
        
        # Find the current step from the most recent response
        if recent_memories:
            latest_response = recent_memories[0].get('bot', '')
            # Extract step number from the response
            import re
            step_match = re.search(r'Step (\d+):', latest_response)
            if step_match:
                current_step = int(step_match.group(1))
        
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
    
    # Check if this is a follow-up question or new topic
    has_conversation_history = bool(memory_str.strip())
    is_question = any(word in user_input.lower() for word in ['how', 'what', 'when', 'where', 'why', 'can', 'could', 'would', 'should', 'is', 'are', 'do', 'does'])
    is_affirmative = user_input.lower().strip() in ['yes', 'y', 'ok', 'okay', 'sure', 'continue', 'next', 'proceed']
    
    if has_conversation_history:
        if is_affirmative:
            if current_step:
                next_step = current_step + 1
                ending_instruction = f"Continue to Step {next_step} in the process. Do NOT repeat Step {current_step} or any previous steps. Progress forward in the workflow."
            else:
                ending_instruction = "Continue to the NEXT step in the process. Do NOT repeat previous steps. Progress forward in the workflow."
        elif is_question:
            ending_instruction = "Provide a helpful response that builds on our previous conversation."
        else:
            ending_instruction = "Continue the conversation naturally, acknowledging the context we've established."
    else:
        if is_question:
            ending_instruction = "Provide a clear and helpful response to the user's question."
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
        "- CRITICAL: When user says 'yes' or similar affirmative responses, progress to the NEXT step in the workflow. Do NOT repeat previous steps.\n"
        "- CRITICAL: Maintain step progression - if you were on Step 13, move to Step 14, not back to Step 11.\n\n"
        f"User Role (if known): {state.get('role', 'unknown')}\n"
        f"Module Context (if applicable): {context}\n"
        f"Current Step: {current_step if current_step else 'Not specified'}\n"
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
    }
    state = rag_flow_executor.invoke(state)
    full_response = state.get("llm_response", "[ERROR: No LLM response]")
    # Store memory after response completes
    store_memory(
        tenant_id=state["tenant_id"],
        user_input=request.input,
        response_text=full_response,
        session_id=state["session_id"],
        module=state["module"]
    )
    return {"response": full_response}





@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/welcome")
async def welcome(module: Optional[str] = Query(None, description="Module name")):
    messages = {
        "Onboarding": "👋 Hi there! I'm here to help you with onboarding. You can ask a question or explore common topics to get started.",
        "Attendance": "👋 Hi there! I'm here to help you with attendance management. You can ask a question or explore common topics to get started.",
        "Payroll": "👋 Hi there! I'm here to help you with payroll services. You can ask a question or explore common topics to get started.",
    }
    suggestions = {
        "Onboarding": [
            "How do I onboard an employee with the mobile app?",
            "What do I need before starting mobile onboarding?",
            "How do I complete web onboarding after mobile onboarding?",
            "Which details and documents are needed in web onboarding?",
            "How do I track the onboarding status?"
        ],
        "Attendance": [
            "How do I mark daily attendance?",
            "How do I request time off?",
            "How can I view my attendance history?",
            "Who do I contact for attendance corrections?"
        ],
        "Payroll": [
            "When will I receive my salary?",
            "How do I view my payslips?",
            "How do I update my bank details?",
            "Who handles payroll queries?"
        ]
    }
    default_message = "Welcome! Here are some suggestions to get started."
    module_key = module or ""
    return {
        "message": messages.get(module_key, default_message),
        "suggestions": suggestions.get(module_key, [
            "How do I get started?",
            "Where can I find help resources?",
            "Who do I contact for support?"
        ])
    }
