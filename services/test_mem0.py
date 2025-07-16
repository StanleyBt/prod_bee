import os
from mem0 import Memory
from dotenv import load_dotenv
import weaviate

load_dotenv()

# Connect to local Weaviate instance
client = weaviate.connect_to_local()

# Create a test collection if it doesn't exist
collection_name = "Mem0Memory"
if not client.collections.exists(collection_name):
    client.collections.create(
        name=collection_name,
        properties=[
            weaviate.classes.config.Property(name="memory", data_type=weaviate.classes.config.DataType.TEXT),
            weaviate.classes.config.Property(name="metadata", data_type=weaviate.classes.config.DataType.TEXT),
            weaviate.classes.config.Property(name="user_id", data_type=weaviate.classes.config.DataType.TEXT),
            weaviate.classes.config.Property(name="created_at", data_type=weaviate.classes.config.DataType.DATE),
        ],
        vectorizer_config=None
    )
    print(f"Created collection: {collection_name}")
else:
    print(f"Collection already exists: {collection_name}")

config = {
    "vector_store": {
        "provider": "weaviate",
        "config": {
            "cluster_url": os.getenv("WEAVIATE_URL", "http://localhost:8080"),
            "collection_name": collection_name,
            "auth_client_secret": None,
        }
    },
    "llm": {
        "provider": "azure_openai",
        "config": {
            "model": "gpt-4.1-mini",
            "temperature": 0.1,
            "max_tokens": 2000,
            "azure_kwargs": {
                  "azure_deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                  "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
                  "azure_endpoint": os.getenv("AZURE_OPENAI_API_BASE"),
                  "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            }
        }
    },
    "embedder": {
        "provider": "azure_openai",
        "config": {
            "model": "text-embedding-ada-002",
            "azure_kwargs": {
                "api_version": os.getenv("AZURE_EMBEDDING_API_VERSION"),
                "azure_deployment": os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME"),
                "azure_endpoint": os.getenv("AZURE_EMBEDDING_API_BASE"),
                "api_key": os.getenv("AZURE_EMBEDDING_API_KEY"),
            }
        }
    }
}

m = Memory.from_config(config)
messages = [
    {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
    {"role": "assistant", "content": "How about a thriller movies? They can be quite engaging."},
    {"role": "user", "content": "I’m not a big fan of thriller movies but I love sci-fi movies."},
    {"role": "assistant", "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future."}
]
add_result = m.add(messages, user_id="test_tenant:test_user2:test_module:test_session")

print("Add result:", add_result)
result = m.get_all(user_id="test_tenant:test_user2:test_module:test_session")
print("Get all result:", result) 


# result = m.get_all(user_id="test_tenant:test_user:test_module:test_session")
# print(result)