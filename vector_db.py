# vector_db.py

import logging
from typing import Optional
from weaviate import WeaviateClient
from weaviate.connect import ConnectionParams, ProtocolParams
from weaviate.collections.classes.config import Configure, Property, DataType
from config import (
    AZURE_EMBEDDING_API_KEY,
    AZURE_EMBEDDING_API_BASE,
    AZURE_EMBEDDING_API_VERSION,
    AZURE_EMBEDDING_DEPLOYMENT_NAME,
)
import openai

logger = logging.getLogger(__name__)

weaviate_client: Optional[WeaviateClient] = None
collection = None

def get_openai_embedding(text):
    try:
        # Check if required config values are available
        if not AZURE_EMBEDDING_API_KEY or not AZURE_EMBEDDING_API_BASE or not AZURE_EMBEDDING_DEPLOYMENT_NAME:
            logger.error("Azure embedding configuration is missing")
            return None
            
        client = openai.AzureOpenAI(
            api_key=AZURE_EMBEDDING_API_KEY,
            api_version=AZURE_EMBEDDING_API_VERSION,
            azure_endpoint=AZURE_EMBEDDING_API_BASE
        )
        response = client.embeddings.create(
            input=[text],
            model=AZURE_EMBEDDING_DEPLOYMENT_NAME
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding call failed: {e}")
        return None

def initialize_weaviate():
    global weaviate_client, collection
    try:
        conn = ConnectionParams(
            http=ProtocolParams(host="localhost", port=8080, secure=False),
            grpc=ProtocolParams(host="localhost", port=50051, secure=False)
        )
        weaviate_client = WeaviateClient(conn)
        weaviate_client.connect()
        if weaviate_client.is_ready():
            version_info = weaviate_client.get_meta()
            logger.info(f"Connected to Weaviate: {version_info.get('version', 'Unknown')}")
            setup_weaviate_collection()
            return True
        else:
            logger.error("Weaviate is not ready")
            return False
    except Exception as e:
        logger.error(f"Weaviate connection failed: {e}")
        return False

def setup_weaviate_collection():
    global collection
    
    # Check if weaviate_client is properly initialized
    if weaviate_client is None:
        logger.error("Weaviate client not initialized")
        return
        
    class_name = "HelloWorld"
    try:
        if not weaviate_client.collections.exists(name=class_name):
            weaviate_client.collections.create(
                name=class_name,
                properties=[Property(name="text", data_type=DataType.TEXT)],
                vectorizer_config=Configure.Vectorizer.none()
            )
            logger.info(f"Collection '{class_name}' created.")
        else:
            logger.info(f"Collection '{class_name}' already exists.")
        collection = weaviate_client.collections.get(class_name)
    except Exception as e:
        logger.error(f"Collection setup failed: {e}")

def store_in_weaviate(user_input, span=None):
    if not (weaviate_client and weaviate_client.is_connected() and collection):
        logger.warning("Weaviate not available for storage")
        return
    try:
        vector = get_openai_embedding(user_input)
        if not vector:
            logger.error("Failed to get embedding for Weaviate storage")
            return
        collection.data.insert(
            properties={"text": user_input},
            vector=vector
        )
        logger.info("Stored in Weaviate.")
        if span:
            try: span.update(metadata={"weaviate_status": "Stored"})
            except Exception: pass
    except Exception as e:
        logger.error(f"Weaviate insert failed: {e}")
        if span:
            try: span.update(metadata={"weaviate_error": str(e)})
            except Exception: pass

def close_weaviate():
    global weaviate_client
    if weaviate_client:
        try:
            weaviate_client.close()
            logger.info("Weaviate client closed.")
        except Exception as e:
            logger.error(f"Weaviate close failed: {e}")
