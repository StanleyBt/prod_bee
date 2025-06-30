import logging
from typing import Dict, Any, Optional, List
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
from langgraph.graph import Graph, END
from services.vector_db import retriever_tool
from services.llm import generate_llm_response
from services.memory_store import store_in_mem0, retrieve_mem0_memories
from core.config import DEFAULT_MODULE
from api.lifespan import lifespan
from tracing import get_langfuse_client
from fastapi import Query

logger = logging.getLogger(__name__)

app = FastAPI(lifespan=lifespan, title="Multi-Tenant RAG API")

# --- MODELS ---
class QueryRequest(BaseModel):
    input: str
    tenant_id: str
    session_id: str
    module: Optional[str] = None
    model_config = ConfigDict(extra="forbid")

class TextResponse(BaseModel):
    response: str

# --- LANGGRAPH RAG FLOW ---
def create_rag_flow():
    graph = Graph()
    langfuse = get_langfuse_client()

    def memory_retrieval_node(state: dict) -> dict:
        """Retrieve conversation memories FIRST"""
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
                    context = "\n\n".join(result) if result else "No context found"
                    state["context"] = context
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
                state["context"] = "\n\n".join(result) if result else "No context found"
            except Exception:
                state["context"] = "Retrieval error"
            return state

    def llm_node(state: dict) -> dict:
        tenant_id = state.get("tenant_id")
        session_id = state.get("session_id")
        module = state.get("module") or DEFAULT_MODULE
        user_input = state["input"]
        memories = state.get("memories", [])
        
        memory_str = "\n".join([
            f"User: {m.get('user', '')}\nAssistant: {m.get('bot', '')}" for m in memories[-3:]
        ]) if memories else ""

        prompt = (
            "You are a smart, friendly, and professional AI assistant built for a modern SaaS platform. "
            "Your job is to help users across different modules of the platform.\n\n"
            "Instructions:\n"
            "- Understand the user's intent using context and conversation history.\n"
            "- Respond clearly and concisely using markdown formatting — include bullet points, numbered steps, or headings if useful.\n"
            "- Avoid overwhelming users — provide just enough detail to address their current need. If the user asks a narrow question, do not give a full process dump.\n"
            "- Adapt your tone to be professional yet approachable.\n"
            "- Respect the user`s role (e.g., employee, manager, vendor, HR) to customize instructions.\n"
            "- Keep responses relevant — don't repeat earlier answers unless needed for clarity.\n"
            "- Support any module (not just onboarding) — your responses should be modular and context-aware.\n"
            "- End each message with a friendly check-in like: 'Would you like help with anything else?' or 'Let me know if you'd like to continue to the next step 😊'.\n\n"
            f"User Role (if known): {state.get('role', 'unknown')}\n"
            f"Module Context (if applicable): {state.get('context', '')}\n"
            f"Conversation History:\n{memory_str}\n\n"
            f"User Input:\n{user_input}\n\n"
            "Respond helpfully and clearly below:"
        )


        if langfuse:
            with langfuse.start_as_current_span(name="LLMGeneration") as span:
                span.update(input={
                    "prompt": prompt,
                    "original_query": user_input,
                    "memory_count": len(memories)
                })
                try:
                    response = generate_llm_response(prompt, span)
                    state["output"] = response
                    span.update(output={"response": response})
                except Exception as e:
                    span.record_exception(e)
                    state["output"] = "Response generation failed"
                return state
        else:
            try:
                response = generate_llm_response(prompt)
                state["output"] = response
            except Exception:
                state["output"] = "Response generation failed"
            return state

    def memory_storage_node(state: dict) -> dict:
        tenant_id = state.get("tenant_id")
        session_id = state.get("session_id")
        module = state.get("module") or DEFAULT_MODULE
        user_input = state["input"]
        output = state.get("output", "")

        store_in_mem0(
            tenant_id, user_input, output, session_id=session_id, module=module
        )
        logger.info(f"Stored new memory for session: {session_id}")
        return state

    graph.add_node("memory_retrieval", memory_retrieval_node)
    graph.add_node("retrieve", retriever_node)
    graph.add_node("generate", llm_node)
    graph.add_node("memory_storage", memory_storage_node)
    
    graph.set_entry_point("memory_retrieval")
    graph.add_edge("memory_retrieval", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "memory_storage")
    graph.add_edge("memory_storage", END)
    return graph.compile()

rag_flow_executor = create_rag_flow()

# --- FastAPI Endpoint ---
@app.post("/query", response_model=TextResponse)
async def query_rag(request: QueryRequest):
    state = {
        "input": request.input,
        "tenant_id": request.tenant_id,
        "session_id": request.session_id,
        "module": request.module or DEFAULT_MODULE,
    }
    state = rag_flow_executor.invoke(state)
    return TextResponse(response=state.get("output", "No response generated"))

@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/welcome")
async def welcome(module: Optional[str] = Query(None, description="Module name")):
    # Module-specific welcome messages
    messages = {
        "Onboarding": "👋 Hi there! I'm here to help you with onboarding. You can ask a question or explore common topics to get started.",
        "Attendance": "👋 Hi there! I'm here to help you with attendance management. You can ask a question or explore common topics to get started.",
        "Payroll": "👋 Hi there! I'm here to help you with payroll services. You can ask a question or explore common topics to get started.",
        # Add other modules as needed
    }
    
    # Module-specific suggested questions
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
    
    # Default message if module not recognized
    default_message = "Welcome! Here are some suggestions to get started."
    
    return {
        "message": messages.get(module, default_message),
        "suggestions": suggestions.get(module, [
            "How do I get started?",
            "Where can I find help resources?",
            "Who do I contact for support?"
        ])
    }