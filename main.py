"""
Enterprise RAG Chatbot Backend
FastAPI + FAISS + Azure OpenAI / OpenAI-compatible embeddings
"""
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import json
import time
import uuid
import logging
from datetime import datetime

from rag_pipeline import RAGPipeline
from document_processor import DocumentProcessor
from chat_memory import ChatMemory
from monitoring import MonitoringService

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ─── App Init ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Enterprise RAG Chatbot API",
    description="Retrieval-Augmented Generation with FAISS + Azure OpenAI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Singletons ──────────────────────────────────────────────────────────────
rag = RAGPipeline()
processor = DocumentProcessor()
memory = ChatMemory()
monitor = MonitoringService()

# ─── Models ──────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    question: str
    top_k: int = 5

class FeedbackRequest(BaseModel):
    session_id: str
    message_id: str
    rating: int          # 1 = thumbs up, -1 = thumbs down
    comment: Optional[str] = None

class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    status: str

class ChatResponse(BaseModel):
    message_id: str
    answer: str
    citations: List[Dict[str, Any]]
    session_id: str
    latency_ms: float
    tokens_used: int

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a PDF, Excel, or text file.
    Text is chunked, embedded, and indexed in FAISS.
    """
    allowed = {".pdf", ".xlsx", ".xls", ".csv", ".txt", ".md", ".docx"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"File type {ext} not supported. Use: {allowed}")

    content = await file.read()
    doc_id = str(uuid.uuid4())

    try:
        chunks = processor.process(content, file.filename, ext)
        rag.add_documents(doc_id, file.filename, chunks)
        logger.info(f"Ingested {file.filename} → {len(chunks)} chunks (id={doc_id})")
        monitor.log_ingest(doc_id, file.filename, len(chunks))

        return IngestResponse(
            doc_id=doc_id,
            filename=file.filename,
            chunk_count=len(chunks),
            status="indexed",
        )
    except Exception as e:
        logger.error(f"Ingest failed for {file.filename}: {e}")
        raise HTTPException(500, f"Ingestion error: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Ask a question. The RAG pipeline:
      1. Embeds the question
      2. Retrieves top-k relevant chunks from FAISS
      3. Builds a prompt with history + context
      4. Calls LLM and returns answer with citations
    """
    t0 = time.time()
    message_id = str(uuid.uuid4())

    history = memory.get(req.session_id)

    try:
        result = rag.query(
            question=req.question,
            session_id=req.session_id,
            chat_history=history,
            top_k=req.top_k,
        )
    except Exception as e:
        logger.error(f"Query failed (session={req.session_id}): {e}")
        raise HTTPException(500, f"Query error: {str(e)}")

    latency_ms = (time.time() - t0) * 1000

    # Persist to chat history
    memory.add(req.session_id, "user", req.question)
    memory.add(req.session_id, "assistant", result["answer"])

    monitor.log_query(
        session_id=req.session_id,
        message_id=message_id,
        question=req.question,
        latency_ms=latency_ms,
        tokens=result.get("tokens_used", 0),
        doc_ids=[c["doc_id"] for c in result["citations"]],
    )

    logger.info(f"Chat answered in {latency_ms:.0f}ms (session={req.session_id})")

    return ChatResponse(
        message_id=message_id,
        answer=result["answer"],
        citations=result["citations"],
        session_id=req.session_id,
        latency_ms=round(latency_ms, 1),
        tokens_used=result.get("tokens_used", 0),
    )


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    """Store user thumbs up/down feedback for a message."""
    monitor.log_feedback(
        session_id=req.session_id,
        message_id=req.message_id,
        rating=req.rating,
        comment=req.comment,
    )
    logger.info(f"Feedback: msg={req.message_id} rating={req.rating}")
    return {"status": "recorded"}


@app.get("/history/{session_id}")
async def get_history(session_id: str):
    """Retrieve conversation history for a session."""
    return {"session_id": session_id, "messages": memory.get(session_id)}


@app.delete("/history/{session_id}")
async def clear_history(session_id: str):
    """Clear conversation history for a session."""
    memory.clear(session_id)
    return {"status": "cleared"}


@app.get("/documents")
async def list_documents():
    """List all ingested documents."""
    return {"documents": rag.list_documents()}


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Remove a document from the index."""
    rag.delete_document(doc_id)
    return {"status": "deleted", "doc_id": doc_id}


@app.get("/metrics")
async def metrics():
    """Monitoring metrics: query counts, latencies, feedback distribution."""
    return monitor.get_metrics()


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
