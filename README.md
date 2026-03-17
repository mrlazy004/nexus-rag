# Nexus RAG — Enterprise Knowledge Assistant

> A production-grade Retrieval-Augmented Generation chatbot built with FastAPI, FAISS, Azure OpenAI, and a React-style single-file frontend.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND (HTML/JS)                      │
│  Upload docs │ Chat UI │ Citations │ Feedback │ Metrics sidebar │
└─────────────────────┬───────────────────────────────────────────┘
                      │ REST API
┌─────────────────────▼───────────────────────────────────────────┐
│                    FASTAPI BACKEND (port 8000)                   │
│                                                                  │
│  /ingest    → DocumentProcessor → chunks → embed → FAISS        │
│  /chat      → embed question → FAISS search → LLM → response    │
│  /feedback  → MonitoringService → feedback.jsonl                │
│  /metrics   → query counts, latency, thumbs up/down             │
└──────────┬─────────────────────┬────────────────────────────────┘
           │                     │
  ┌────────▼───────┐   ┌─────────▼──────────────────────┐
  │  FAISS Index   │   │     Azure OpenAI                │
  │  (in-memory)   │   │  • text-embedding-ada-002       │
  │  cosine sim    │   │  • gpt-4o (chat completion)     │
  └────────────────┘   └─────────────────────────────────┘
```

---

## RAG Pipeline Explained

### 1. Document Ingestion
```
File upload → DocumentProcessor → text chunks (800 tok, 150 overlap)
                                → Azure OpenAI embeddings (ada-002)
                                → FAISS IndexFlatIP (cosine similarity)
```
Supported formats: **PDF** (text + tables), **Excel** (sheet-by-sheet), **CSV**, **TXT/MD**, **DOCX**.

Chunking strategy: paragraph-aware recursive split with configurable size and overlap to preserve context at boundaries.

### 2. Retrieval
```
User question → embed with ada-002 → L2-normalize
               → FAISS inner-product search (top-k=5)
               → ranked chunks with relevance scores
```

### 3. Augmented Generation
```
Chunks + chat history (last 6 turns) → prompt template
  → gpt-4o (temp=0.1 for factual accuracy)
  → answer with [Source N] inline citations
```

### 4. Prompt Engineering
The system prompt enforces:
- Citation of every claim with `[Source N]`
- No fabrication — answer only from context
- Exact numerical data (financials, KPIs)
- Conservative policy interpretation
- Structured output with optional Summary

---

## Project Structure

```
rag-chatbot/
├── backend/
│   ├── main.py              # FastAPI app, all routes
│   ├── rag_pipeline.py      # FAISS store + Azure OpenAI RAG
│   ├── document_processor.py # PDF/Excel/CSV/TXT/DOCX parsing
│   ├── chat_memory.py       # In-memory session history
│   ├── monitoring.py        # Metrics + JSONL logging
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   └── index.html           # Full React-style SPA (no build step)
├── infra/
│   └── main.bicep           # Azure Container Apps + OpenAI IaC
├── .github/workflows/
│   └── deploy.yml           # CI/CD pipeline
├── Dockerfile               # Backend container
├── docker-compose.yml       # Local dev
└── README.md
```

---

## Quick Start (Local)

### Prerequisites
- Python 3.11+
- Azure OpenAI resource with `text-embedding-ada-002` and `gpt-4o` deployed
  (or standard OpenAI API key)

### 1. Backend

```bash
cd backend
cp .env.example .env
# Fill in your Azure OpenAI keys in .env

pip install -r requirements.txt
mkdir logs
uvicorn main:app --reload --port 8000
```

### 2. Frontend

```bash
# Serve with any static server
cd frontend
npx serve .          # or python -m http.server 3000
```

Open `http://localhost:3000` in your browser.

### 3. Docker Compose (recommended)

```bash
# Fill backend/.env first
docker compose up --build
```

- Backend: http://localhost:8000
- Frontend: http://localhost:3000
- API docs: http://localhost:8000/docs

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ingest` | Upload a file (form-data `file`) |
| `POST` | `/chat` | Ask a question |
| `POST` | `/feedback` | Submit thumbs up/down |
| `GET` | `/history/{session_id}` | Get conversation history |
| `DELETE` | `/history/{session_id}` | Clear conversation |
| `GET` | `/documents` | List indexed documents |
| `DELETE` | `/documents/{doc_id}` | Remove a document |
| `GET` | `/metrics` | Query/feedback metrics |
| `GET` | `/health` | Health check |

### Chat request
```json
POST /chat
{
  "session_id": "abc123",
  "question": "What were Q3 sales?",
  "top_k": 5
}
```

### Chat response
```json
{
  "message_id": "uuid",
  "answer": "Q3 sales were $4.2M [Source 1]...",
  "citations": [
    {
      "source_num": 1,
      "filename": "Q3_report.pdf",
      "page": 12,
      "excerpt": "Total Q3 revenue: $4.2M...",
      "relevance_score": 0.923
    }
  ],
  "latency_ms": 1240.5,
  "tokens_used": 892
}
```

---

## Azure Deployment

### 1. Create resources

```bash
az group create -n nexus-rag-rg -l eastus

az deployment group create \
  -g nexus-rag-rg \
  --template-file infra/main.bicep \
  --parameters appName=nexus-rag
```

### 2. Create ACR and push images

```bash
ACR=youruniquename
az acr create -g nexus-rag-rg -n $ACR --sku Basic
az acr login -n $ACR

docker build -f Dockerfile -t $ACR.azurecr.io/nexus-rag-backend:latest ./backend
docker push $ACR.azurecr.io/nexus-rag-backend:latest
```

### 3. Configure GitHub secrets (for CI/CD)

```
AZURE_CREDENTIALS  → az ad sp create-for-rbac --sdk-auth output
```

Push to `main` branch → GitHub Actions builds, pushes to ACR, deploys Bicep, runs smoke test.

### 4. Update CORS for production

In `main.py`, change:
```python
allow_origins=["https://your-frontend.azurecontainerapps.io"]
```

---

## Production Hardening Checklist

- [ ] Replace in-memory FAISS with **Azure Cognitive Search** for persistence + scale
- [ ] Replace in-memory ChatMemory with **Azure Cache for Redis**
- [ ] Store JSONL logs in **Azure Blob Storage** or **Application Insights**
- [ ] Add **Azure AD / Entra ID** authentication (MSAL)
- [ ] Enable **Azure Key Vault** for secrets (not env vars)
- [ ] Set CORS to specific frontend domain
- [ ] Enable **HTTPS** (automatic with Container Apps)
- [ ] Configure **Azure Front Door** for CDN + WAF
- [ ] Set up **budget alerts** on Azure OpenAI token usage
- [ ] Add rate limiting (slowapi or APIM)

---

## Swap FAISS → Azure Cognitive Search

Replace the `FAISSStore` class in `rag_pipeline.py`:

```python
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential

class AzureCognitiveSearchStore:
    def __init__(self):
        self.client = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name="rag-index",
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
        )
    
    def add(self, chunks):
        docs = [{"id": c.chunk_id, "content": c.text,
                 "contentVector": c.embedding.tolist(),
                 "filename": c.filename, "page": c.page}
                for c in chunks]
        self.client.upload_documents(docs)
    
    def search(self, query_embedding, top_k=5):
        results = self.client.search(
            search_text="",
            vector_queries=[VectorizedQuery(
                vector=query_embedding.tolist(),
                k_nearest_neighbors=top_k,
                fields="contentVector"
            )]
        )
        return [(r, r["@search.score"]) for r in results]
```

---

## Monitoring

Logs are written to `logs/` as JSONL:
- `queries.jsonl` — every question with latency + tokens
- `feedback.jsonl` — user thumbs up/down
- `ingests.jsonl` — document ingestion events

In production, forward these to **Azure Monitor** via the Container Apps built-in log drain.

---

## License

MIT
