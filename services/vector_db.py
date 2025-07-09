import logging
from typing import List, Dict, Optional
import weaviate
from weaviate.classes.query import Filter
from weaviate.classes.config import Configure, Property, DataType  # ✅ Correct import
from core.config import WEAVIATE_URL, WEAVIATE_COLLECTION_NAME
import openai
from core.config import (
    AZURE_EMBEDDING_API_KEY,
    AZURE_EMBEDDING_API_BASE,
    AZURE_EMBEDDING_API_VERSION,
    AZURE_EMBEDDING_DEPLOYMENT_NAME
)

logger = logging.getLogger(__name__)

weaviate_client: Optional[weaviate.Client] = None
collection_name = WEAVIATE_COLLECTION_NAME

def initialize_weaviate() -> bool:
    global weaviate_client
    try:
        weaviate_client = weaviate.connect_to_local()
        if weaviate_client.is_ready():
            version_info = weaviate_client.get_meta()
            logger.info(f"Connected to Weaviate version: {version_info.get('version', 'Unknown')}")
            return True
        else:
            logger.error("Weaviate is not ready")
            return False
    except Exception as e:
        logger.error(f"Weaviate connection failed: {e}")
        return False

def get_openai_embedding(text: str) -> Optional[List[float]]:
    try:
        if not AZURE_EMBEDDING_API_KEY or not AZURE_EMBEDDING_API_BASE or not AZURE_EMBEDDING_DEPLOYMENT_NAME:
            logger.error("Azure embedding configuration is missing")
            return None

        client = openai.AzureOpenAI(
            api_key=AZURE_EMBEDDING_API_KEY,
            api_version=AZURE_EMBEDDING_API_VERSION,
            azure_endpoint=AZURE_EMBEDDING_API_BASE,
        )
        response = client.embeddings.create(
            input=[text],
            model=AZURE_EMBEDDING_DEPLOYMENT_NAME
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding call failed: {e}")
        return None

def collection_exists() -> bool:
    if weaviate_client:
        try:
            return weaviate_client.collections.exists(collection_name)
        except Exception as e:
            logger.error(f"Failed to check collection existence: {e}")
    return False

def create_collection_schema() -> bool:
    global weaviate_client
    if not weaviate_client:
        logger.error("Weaviate client not initialized")
        return False
    try:
        if not weaviate_client.collections.exists(collection_name):
            weaviate_client.collections.create(
                name=collection_name,
                multi_tenancy_config=Configure.multi_tenancy(
                    enabled=True,
                    auto_tenant_creation=True
                ),
                properties=[
                    Property(name="tenant_id", data_type=DataType.TEXT),
                    Property(name="module", data_type=DataType.TEXT),
                    Property(name="document_name", data_type=DataType.TEXT),
                    Property(name="text", data_type=DataType.TEXT),
                ],
                vectorizer_config=None
            )
            logger.info(f"Collection '{collection_name}' created with multi-tenancy enabled.")
        else:
            logger.info(f"Collection '{collection_name}' already exists.")
        return True
    except Exception as e:
        logger.error(f"Collection setup failed: {e}")
        return False

def store_chunks_batch(chunks: List[Dict[str, str]]) -> None:
    global weaviate_client
    if not weaviate_client:
        logger.warning("Weaviate not available for storage")
        return

    try:
        collection = weaviate_client.collections.get(collection_name)
        for chunk in chunks:
            text = chunk.get("text", "")
            tenant_id = chunk.get("tenant_id")
            module = chunk.get("module")
            document_name = chunk.get("document_name")

            if not text or not tenant_id or not module or not document_name:
                logger.warning("Invalid chunk data: missing required metadata")
                continue

            vector = get_openai_embedding(text)
            if not vector:
                logger.error("Failed to get embedding for chunk")
                continue

            tenant_collection = collection.with_tenant(tenant_id)
            tenant_collection.data.insert(
                properties={
                    "tenant_id": tenant_id,
                    "module": module,
                    "document_name": document_name,
                    "text": text,
                },
                vector=vector
            )
        logger.info(f"Batch inserted {len(chunks)} chunks into Weaviate.")
    except Exception as e:
        logger.error(f"Weaviate batch insert failed: {e}")

def get_available_documents(tenant_id: str, module: str) -> List[Dict[str, str]]:
    if not weaviate_client:
        logger.warning("Weaviate not available for document listing")
        return []

    try:
        collection = weaviate_client.collections.get(collection_name)
        result = collection.aggregate.over_properties(["document_name"]) \
            .with_where({
                "operator": "And",
                "operands": [
                    {"path": ["tenant_id"], "operator": "Equal", "valueString": tenant_id},
                    {"path": ["module"], "operator": "Equal", "valueString": module}
                ]
            }) \
            .with_fields("document_name {count}") \
            .do()

        docs = []
        for agg in result.get("data", {}).get("Aggregate", {}).get(collection_name, []):
            doc_name = agg.get("document_name")
            if doc_name:
                docs.append({"label": doc_name, "value": doc_name})
        return docs
    except Exception as e:
        logger.error(f"Failed to list available documents: {e}")
        return []

def retriever_tool(
    query: str,
    tenant_id: str,
    module: str,
    top_k: int = 3
) -> List[str]:
    global weaviate_client
    if not weaviate_client:
        logger.error("Weaviate client not connected")
        return []

    query_embedding = get_openai_embedding(query)
    if not query_embedding:
        logger.error("Failed to generate embedding for query")
        return []

    try:
        filters = Filter.by_property("tenant_id").equal(tenant_id)
        filters = filters & Filter.by_property("module").equal(module)

        collection = weaviate_client.collections.get(collection_name)
        results = collection.with_tenant(tenant_id).query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            filters=filters
        ).objects

        texts = [obj.properties["text"] for obj in results]
        logger.info(f"Retriever found {len(texts)} results for query: {query}")
        return texts

    except Exception as e:
        logger.error(f"Retriever query failed: {e}", exc_info=True)
        return []

def close_weaviate() -> None:
    global weaviate_client
    if weaviate_client:
        try:
            weaviate_client.close()
            logger.info("Weaviate client closed.")
        except Exception as e:
            logger.error(f"Weaviate close failed: {e}")

def delete_collection() -> None:
    """
    Delete all collections from Weaviate to start fresh.
    """
    global weaviate_client
    if not weaviate_client:
        logger.error("Weaviate client not initialized")
        return

    try:
        schema = weaviate_client.schema.get()
        classes = schema.get("classes", [])
        for cls in classes:
            class_name = cls.get("class")
            logger.info(f"Deleting collection '{class_name}'")
            weaviate_client.collections.delete(class_name)
        logger.info("All collections deleted.")
    except Exception as e:
        logger.error(f"Failed to delete collections: {e}")
