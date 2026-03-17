"""
Chat Memory - in-process session store.
For production: swap backend with Redis or Azure Cache.
"""

from typing import List, Dict, Optional
from collections import defaultdict
import threading

MAX_HISTORY = 50   # messages per session before rolling


class ChatMemory:

    def __init__(self):
        self._store: Dict[str, List[Dict]] = defaultdict(list)
        self._lock = threading.Lock()

    def add(self, session_id: str, role: str, content: str):
        with self._lock:
            msgs = self._store[session_id]
            msgs.append({"role": role, "content": content})
            # Rolling window
            if len(msgs) > MAX_HISTORY:
                self._store[session_id] = msgs[-MAX_HISTORY:]

    def get(self, session_id: str) -> List[Dict]:
        with self._lock:
            return list(self._store.get(session_id, []))

    def clear(self, session_id: str):
        with self._lock:
            self._store.pop(session_id, None)
