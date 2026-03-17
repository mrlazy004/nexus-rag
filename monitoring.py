"""
Monitoring & Logging Service
Tracks: query latency, token usage, feedback, document ingestion.
Persists to JSON log files. Replace with Azure Monitor / App Insights in prod.
"""

import json
import os
import threading
from datetime import datetime
from typing import List, Optional, Dict, Any
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)
LOG_DIR = "logs"


class MonitoringService:

    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        self._lock = threading.Lock()
        self._queries: List[Dict] = []
        self._feedback: List[Dict] = []
        self._ingests: List[Dict] = []

    def _append_log(self, filename: str, record: Dict):
        path = os.path.join(LOG_DIR, filename)
        with self._lock:
            with open(path, "a") as f:
                f.write(json.dumps(record) + "\n")

    def log_query(
        self,
        session_id: str,
        message_id: str,
        question: str,
        latency_ms: float,
        tokens: int,
        doc_ids: List[str],
    ):
        record = {
            "ts": datetime.utcnow().isoformat(),
            "type": "query",
            "session_id": session_id,
            "message_id": message_id,
            "question": question[:200],
            "latency_ms": round(latency_ms, 1),
            "tokens": tokens,
            "source_docs": doc_ids,
        }
        self._queries.append(record)
        self._append_log("queries.jsonl", record)

    def log_feedback(
        self,
        session_id: str,
        message_id: str,
        rating: int,
        comment: Optional[str],
    ):
        record = {
            "ts": datetime.utcnow().isoformat(),
            "type": "feedback",
            "session_id": session_id,
            "message_id": message_id,
            "rating": rating,
            "comment": comment,
        }
        self._feedback.append(record)
        self._append_log("feedback.jsonl", record)
        logger.info(f"Feedback stored: msg={message_id} rating={rating}")

    def log_ingest(self, doc_id: str, filename: str, chunk_count: int):
        record = {
            "ts": datetime.utcnow().isoformat(),
            "type": "ingest",
            "doc_id": doc_id,
            "filename": filename,
            "chunk_count": chunk_count,
        }
        self._ingests.append(record)
        self._append_log("ingests.jsonl", record)

    def get_metrics(self) -> Dict[str, Any]:
        queries = self._queries
        feedback = self._feedback

        avg_latency = (
            sum(q["latency_ms"] for q in queries) / len(queries) if queries else 0
        )
        total_tokens = sum(q["tokens"] for q in queries)
        thumbs_up = sum(1 for f in feedback if f["rating"] == 1)
        thumbs_down = sum(1 for f in feedback if f["rating"] == -1)

        return {
            "total_queries": len(queries),
            "total_ingests": len(self._ingests),
            "avg_latency_ms": round(avg_latency, 1),
            "total_tokens_used": total_tokens,
            "feedback": {
                "thumbs_up": thumbs_up,
                "thumbs_down": thumbs_down,
                "total": len(feedback),
            },
        }
