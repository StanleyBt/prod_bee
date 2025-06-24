# main.py

import logging
from llm import test_llm_connection, generate_llm_response
from vector_db import initialize_weaviate, store_in_weaviate, close_weaviate
from memory_store import initialize_mem0, store_in_mem0, retrieve_mem0_memories
from tracing import get_langfuse_client
from langgraph.graph import END, Graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def hello_node(state):
    user_input = state["input"]
    logger.info(f"Processing user input: {user_input}")
    langfuse = get_langfuse_client()
    if langfuse:
        try:
            with langfuse.start_as_current_span(name="HelloNodeSpan") as span:
                span.update(input=user_input)
                response_text = generate_llm_response(user_input, span)
                store_in_weaviate(user_input, span)
                store_in_mem0(user_input, response_text, span)
                span.update(output={"llm_response": response_text})
                return {"output": response_text}
        except Exception as e:
            logger.error(f"Langfuse span execution failed: {e}")
            return execute_without_span(user_input)
    else:
        return execute_without_span(user_input)

def execute_without_span(user_input):
    response_text = generate_llm_response(user_input)
    store_in_weaviate(user_input)
    store_in_mem0(user_input, response_text)
    return {"output": response_text}

def create_langgraph():
    graph = Graph()
    graph.add_node("hello_node", hello_node)
    graph.set_entry_point("hello_node")
    graph.add_edge("hello_node", END)
    return graph.compile()

def main():
    logger.info("Starting Modularized Mem0 + Weaviate + LangGraph Integration")
    llm_ready = test_llm_connection()
    weaviate_ready = initialize_weaviate()
    mem0_ready = initialize_mem0()

    if not llm_ready:
        logger.error("Cannot proceed without LLM connection")
        return
    if not weaviate_ready:
        logger.warning("Proceeding without Weaviate")
    if not mem0_ready:
        logger.warning("Proceeding without Mem0")

    app_flow = create_langgraph()
    user_input = "Hello from LangGraph + Weaviate + Mem0"
    try:
        result = app_flow.invoke({"input": user_input})
        logger.info(f"Final output: {result['output']}")
        retrieve_mem0_memories()
    except Exception as e:
        logger.error(f"LangGraph execution failed: {e}")

    langfuse = get_langfuse_client()
    if langfuse:
        try:
            langfuse.flush()
            logger.info("Langfuse flushed.")
        except Exception as e:
            logger.error(f"Langfuse flush failed: {e}")
    close_weaviate()

if __name__ == "__main__":
    main()
