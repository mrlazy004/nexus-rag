from dotenv import load_dotenv
load_dotenv()

import os
import uuid
import numpy as np
import faiss
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from groq import Groq
import logging

logger = logging.getLogger(__name__)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
CHAT_MODEL = "llama-3.3-70b-versatile"
EMBED_DIM = 384

def simple_embed(texts):
    result = []
    for text in texts:
        vec = np.zeros(EMBED_DIM)
        words = text.lower().split()
        for word in words:
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vec[h % EMBED_DIM] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        result.append(vec)
    return np.array(result, dtype=np.float32)

SYSTEM_PROMPT = """You are an expert enterprise business assistant.
Answer questions accurately using ONLY the provided context documents.
Rules:
1. Base EVERY claim on the supplied context. Never fabricate data.
2. If the context is insufficient, say so explicitly.
3. Cite each fact with [Source N] inline.
4. Keep answers clear and structured."""

QUERY_PROMPT_TEMPLATE = """
## Context Documents
{context}

## Conversation History
{history}

## Question
{question}

Answer using ONLY the context above. Cite evidence as [Source 1], [Source 2], etc.
If you cannot answer from the context, say "I don't have enough information in the provided documents."
"""

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    filename: str
    text: str
    page: Optional[int] = None
    sheet: Optional[str] = None
    embedding: Optional[np.ndarray] = None

class FAISSStore:
    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: List[Chunk] = []
        self._doc_offsets: Dict[str, List[int]] = {}

    def add(self, chunks: List[Chunk]):
        if not chunks:
            return
        embeddings = np.array([c.embedding for c in chunks], dtype=np.float32)
        faiss.normalize_L2(embeddings)
        start_id = len(self.chunks)
        self.index.add(embeddings)
        for i, chunk in enumerate(chunks):
            self.chunks.append(chunk)
            self._doc_offsets.setdefault(chunk.doc_id, []).append(start_id + i)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        q = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        scores, ids = self.index.search(q, min(top_k, max(1, len(self.chunks))))
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx >= 0 and self.chunks[idx] is not None:
                results.append((self.chunks[idx], float(score)))
        return results

    def delete(self, doc_id: str):
        ids = self._doc_offsets.pop(doc_id, [])
        for i in ids:
            self.chunks[i] = None

    def list_docs(self) -> List[str]:
        return list(self._doc_offsets.keys())

class RAGPipeline:

    def __init__(self):
        self.store = FAISSStore()
        self._doc_meta: Dict[str, Dict] = {}

    def _embed(self, texts: List[str]) -> np.ndarray:
        return simple_embed(texts)

    def add_documents(self, doc_id: str, filename: str, raw_chunks: List[Dict]):
        texts = [c["text"] for c in raw_chunks]
        embeddings = self._embed(texts)
        chunks = [
            Chunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                filename=filename,
                text=rc["text"],
                page=rc.get("page"),
                sheet=rc.get("sheet"),
                embedding=embeddings[i],
            )
            for i, rc in enumerate(raw_chunks)
        ]
        self.store.add(chunks)
        self._doc_meta[doc_id] = {"filename": filename, "chunk_count": len(chunks)}
        logger.info(f"Indexed {len(chunks)} chunks for {filename}")

    def query(self, question: str, session_id: str, chat_history: List[Dict], top_k: int = 5) -> Dict[str, Any]:
        q_emb = self._embed([question])[0]
        results = self.store.search(q_emb, top_k=top_k)
        results = [(c, s) for c, s in results if c is not None]

        if not results:
            return {
                "answer": "I don't have any indexed documents to answer from. Please upload documents first.",
                "citations": [],
                "tokens_used": 0,
            }

        context_parts = []
        citations = []
        for i, (chunk, score) in enumerate(results, 1):
            loc = f"p.{chunk.page}" if chunk.page else (chunk.sheet or "")
            context_parts.append(f"[Source {i}] {chunk.filename} {loc}\n{chunk.text}")
            citations.append({
                "source_num": i,
                "doc_id": chunk.doc_id,
                "filename": chunk.filename,
                "page": chunk.page,
                "sheet": chunk.sheet,
                "excerpt": chunk.text[:200] + ("..." if len(chunk.text) > 200 else ""),
                "relevance_score": round(score, 4),
            })

        context_str = "\n\n---\n\n".join(context_parts)
        history_str = self._format_history(chat_history[-12:])

        user_content = QUERY_PROMPT_TEMPLATE.format(
            context=context_str,
            history=history_str or "None",
            question=question,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        resp = groq_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=1200,
        )

        answer = resp.choices[0].message.content
        tokens_used = resp.usage.total_tokens if resp.usage else 0

        return {
            "answer": answer,
            "citations": citations,
            "tokens_used": tokens_used,
        }

    def _format_history(self, history: List[Dict]) -> str:
        return "\n".join(
            f"{msg.get('role','user').capitalize()}: {msg.get('content','')}"
            for msg in history
        )

    def list_documents(self) -> List[Dict]:
        return [
            {"doc_id": k, **v}
            for k, v in self._doc_meta.items()
            if k in self.store.list_docs()
        ]

    def delete_document(self, doc_id: str):
        self.store.delete(doc_id)
        self._doc_meta.pop(doc_id, None)