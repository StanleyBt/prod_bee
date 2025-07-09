import logging
from typing import Optional, List
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
from langgraph.graph import Graph, END
from services.vector_db import retriever_tool
from services.llm import generate_llm_response
from services.memory_store import store_in_mem0, retrieve_mem0_memories
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
        tenant_id = state.get("tenant_id")
        session_id = state.get("session_id")
        module = state.get("module") or DEFAULT_MODULE

        memories = retrieve_mem0_memories(
            tenant_id=tenant_id,
            session_id=session_id,
            module=module
        )
        state["memories"] = memories
        logger.info(f"Retrieved {len(memories)} memories for session: {session_id}")
        return state

    def retriever_node(state: dict) -> dict:
        tenant_id = state.get("tenant_id")
        module = state.get("module") or DEFAULT_MODULE
        user_input = state["input"]

        def preprocess_context(chunks: List[str], max_chars: int = 1000) -> str:
            combined = " ".join(chunks)
            if len(combined) > max_chars:
                combined = combined[:max_chars] + "..."
            return combined

        if langfuse:
            with langfuse.start_as_current_span(name="Retriever") as span:
                span.update(input={
                    "query": user_input,
                    "tenant_id": tenant_id,
                    "module": module
                })
                try:
                    result = retriever_tool(
                        user_input,
                        tenant_id=tenant_id,
                        module=module,
                        top_k=3
                    )
                    context_summary = preprocess_context(result)
                    state["context"] = context_summary if context_summary else "No context found"
                    span.update(output={"result_count": len(result) if result else 0})
                except Exception as e:
                    span.update(exception=e)
                    state["context"] = "Retrieval error"
                return state
        else:
            try:
                result = retriever_tool(
                    user_input,
                    tenant_id=tenant_id,
                    module=module,
                    top_k=3
                )
                context_summary = preprocess_context(result)
                state["context"] = context_summary if context_summary else "No context found"
            except Exception:
                state["context"] = "Retrieval error"
            return state

    # We do NOT include llm_node here since streaming is handled outside graph
    # Similarly, memory_storage_node is handled after streaming completes

    graph.add_node("memory_retrieval", memory_retrieval_node)
    graph.add_node("retrieve", retriever_node)

    graph.set_entry_point("memory_retrieval")
    graph.add_edge("memory_retrieval", "retrieve")
    graph.add_edge("retrieve", END)
    return graph.compile()

rag_flow_executor = create_rag_flow()

# --- PROMPT BUILDER ---
def build_prompt(state: dict, user_input: str) -> str:
    memories = state.get("memories", [])
    memory_str = "\n".join([
        f"User: {m.get('user', '')}\nAssistant: {m.get('bot', '')}" for m in memories[-3:]
    ]) if memories else ""

    prompt = (
        "You are a smart, friendly, and professional AI assistant built for a modern SaaS platform. "
        "Instructions:\n"
        "- Understand the user's intent using context and conversation history.\n"
        "- Respond clearly and concisely using markdown formatting — include bullet points, numbered steps, or headings if useful.\n"
        "- Avoid overwhelming users — provide just enough detail to address their current need.\n"
        "- Adapt your tone to be professional yet approachable.\n"
        "- Respect the user`s role (e.g., employee, manager, vendor, HR) to customize instructions.\n"
        "- Keep responses relevant — don't repeat earlier answers unless needed for clarity.\n"
        "- Your responses should be modular and context-aware.\n"
        "- IMPORTANT: If the module context does not contain relevant information, please inform the user rather than guessing.\n"
        "- Focus your responses strictly on the user's selected module and avoid discussing other modules.\n"
        "- End each message with a friendly check-in like: 'Would you like help with anything else?' or 'Let me know if you'd like to continue to the next step 😊'.\n\n"
        f"User Role (if known): {state.get('role', 'unknown')}\n"
        f"Module Context (if applicable): {state.get('context', '')}\n"
        f"Conversation History:\n{memory_str}\n\n"
        f"User Input:\n{user_input}\n\n"
        "Respond helpfully and clearly below:"
    )
    return prompt

# --- FASTAPI ENDPOINTS ---
@app.post("/query")
async def query_rag(request: QueryRequest):
    # Run graph nodes up to retrieval
    state = {
        "input": request.input,
        "tenant_id": request.tenant_id,
        "session_id": request.session_id,
        "module": request.module or DEFAULT_MODULE,
    }
    state = rag_flow_executor.invoke(state)

    prompt = build_prompt(state, request.input)

    langfuse = get_langfuse_client()
    response_chunks = []

    def llm_streamer():
        if langfuse:
            with langfuse.start_as_current_span(name="LLMGeneration") as span:
                span.update(input={"prompt": prompt, "original_query": request.input})
                try:
                    for chunk in generate_llm_response(prompt, span):
                        response_chunks.append(chunk)
                        yield chunk  # Yield only actual tokens to client
                    # Update LangFuse internally, do NOT yield this
                    span.update(output={"status": "completed"})
                except Exception as e:
                    span.record_exception(e)
                    yield "[ERROR: Response generation failed]"
        else:
            try:
                for chunk in generate_llm_response(prompt):
                    response_chunks.append(chunk)
                    yield chunk
            except Exception:
                yield "[ERROR: Response generation failed]"

    # Store memory after streaming completes
    full_response = "".join(response_chunks)
    store_in_mem0(
        tenant_id=state["tenant_id"],
        user_input=request.input,
        response_text=full_response,
        session_id=state["session_id"],
        module=state["module"]
    )

    return StreamingResponse(llm_streamer(), media_type="text/plain")


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
    return {
        "message": messages.get(module, default_message),
        "suggestions": suggestions.get(module, [
            "How do I get started?",
            "Where can I find help resources?",
            "Who do I contact for support?"
        ])
    }
