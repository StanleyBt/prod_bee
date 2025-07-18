# Multi-Tenant RAG Chatbot System

A production-ready, multi-tenant RAG-based chatbot system using Weaviate for vector storage, Mem0 for conversation memory, and Azure OpenAI for LLM and embeddings.

## Features

- **Multi-Tenant Architecture**: Uses Weaviate's built-in multi-tenancy for complete tenant isolation
- **RAG (Retrieval-Augmented Generation)**: Semantic search over tenant-specific documents
- **Conversation Memory**: Persistent conversation history using Mem0
- **Azure OpenAI Integration**: GPT-4 and embedding models via Azure OpenAI
- **FastAPI API**: RESTful API with streaming responses
- **Langfuse Tracing**: Comprehensive observability and performance monitoring
- **PDF Processing**: Automatic document ingestion and chunking
- **Module-Based Organization**: Tenant data organized by modules (e.g., Attendance, Onboarding)

## Architecture

### Per-Tenant Collections

The system uses separate collections for each tenant:

- **Data Collections**: `Documents_{tenant_id}` for each tenant
- **Memory Collections**: `Memory_{tenant_id}` for each tenant
- **Automatic Collection Creation**: New tenant collections are created on first use
- **Complete Isolation**: Each tenant's data is in separate collections

### Key Components

- **API Layer** (`api/`): FastAPI application with streaming responses
- **Core Logic** (`core/`): Configuration, ingestion, and main application logic
- **Services** (`services/`): Vector DB, memory store, and LLM services
- **Utilities** (`utils/`): PDF processing and tenant management tools

## Setup

### Prerequisites

- Python 3.8+
- Weaviate (local or cloud)
- Azure OpenAI account
- Docker (for local Weaviate)

### Environment Variables

Create a `.env` file with:

```env
# Azure OpenAI
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_API_BASE=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment
AZURE_API_VERSION=2025-01-01-preview

# Azure OpenAI Embeddings
AZURE_EMBEDDING_API_KEY=your_key
AZURE_EMBEDDING_API_BASE=https://your-resource.openai.azure.com/
AZURE_EMBEDDING_DEPLOYMENT_NAME=your_embedding_deployment
AZURE_EMBEDDING_API_VERSION=2023-05-15

# Weaviate
WEAVIATE_URL=http://localhost:8080
WEAVIATE_COLLECTION_NAME=multi_tenant_collection

# Langfuse (optional)
LANGFUSE_PUBLIC_KEY=your_key
LANGFUSE_SECRET_KEY=your_secret
LANGFUSE_HOST=https://cloud.langfuse.com
```

### Installation

1. **Clone and setup**:
   ```bash
   git clone <repository>
   cd prod-bee
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Start Weaviate**:
   ```bash
   docker-compose up -d
   ```

3. **Add documents**:
   Place PDF files in `data/{tenant_id}/{module}/` structure:
   ```
   data/
   ├── CWFM/
   │   ├── Attendance/
   │   │   └── Staff Guide.pdf
   │   └── Onboarding/
   │       ├── Employee Mobile Onboarding.pdf
   │       └── Vendor Web Onboarding.pdf
   └── ANOTHER_TENANT/
       └── General/
           └── Company Policy.pdf
   ```

4. **Run the application**:
   ```bash
   python -m core.main
   ```

## Usage

### API Endpoints

- `POST /query`: Main chat endpoint
  ```json
  {
    "tenant_id": "CWFM",
    "module": "Attendance", 
    "session_id": "user-123",
    "message": "How do I request time off?"
  }
  ```

### Per-Tenant Features

- **Automatic Collection Creation**: New tenant collections are created automatically when first accessed
- **Complete Tenant Isolation**: Each tenant's data and conversations are in separate collections
- **Module Organization**: Data is organized by modules within each tenant's collections
- **Session Management**: Conversation memory is maintained per tenant/module/session

### Tenant Management

Use the per-tenant storage utilities:

```python
from services.per_tenant_storage import list_tenant_collections, ensure_tenant_collections

# List all tenant collections
collections = list_tenant_collections()

# Ensure collections exist for a new tenant
success = ensure_tenant_collections("NEW_TENANT")
```

## Development

### Testing

Run the per-tenant ingestion test:
```bash
python test_per_tenant_ingestion.py
```

### Adding New Tenants

1. Create tenant directory: `data/{tenant_id}/`
2. Add modules: `data/{tenant_id}/{module}/`
3. Add PDF documents
4. The system will automatically create the necessary collections

### Monitoring

- **Langfuse Tracing**: View traces at https://cloud.langfuse.com
- **Application Logs**: Detailed logging for debugging
- **Performance Metrics**: Track retrieval and generation performance

## Troubleshooting

### Common Issues

1. **Mem0 Empty Results**: Check if the Mem0Memory collection exists with multi-tenancy enabled
2. **Weaviate Connection**: Ensure Weaviate is running and accessible
3. **Azure OpenAI**: Verify API keys and deployment names
4. **Memory Storage**: Check fallback storage logs for memory issues

### Debug Mode

Enable detailed logging by setting log level to DEBUG in the application.

## License

[Add your license information here]
