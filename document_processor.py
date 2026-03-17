"""
Document Processor
Handles PDF, Excel, CSV, TXT, DOCX ingestion.
Outputs clean text chunks with metadata.
"""

import io
import re
from typing import List, Dict, Optional

# PDF
import pdfplumber

# Excel / CSV
import pandas as pd

# DOCX
from docx import Document as DocxDocument


# ─── Config ──────────────────────────────────────────────────────────────────

CHUNK_SIZE = 800         # target tokens per chunk (approx by chars / 4)
CHUNK_OVERLAP = 150      # overlap to preserve context at boundaries


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Recursive paragraph-aware text splitter.
    Priority: split on double-newline → single newline → sentence → character.
    """
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not text:
        return []

    # If short enough, return as-is
    if len(text) // 4 <= chunk_size:
        return [text]

    # Try splitting on paragraph breaks first
    for sep in ["\n\n", "\n", ". ", " "]:
        parts = text.split(sep)
        if len(parts) > 1:
            chunks = []
            current = ""
            for part in parts:
                candidate = current + sep + part if current else part
                if len(candidate) // 4 <= chunk_size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current.strip())
                    current = part
            if current:
                chunks.append(current.strip())

            # Apply overlap
            overlapped = []
            for i, chunk in enumerate(chunks):
                if i > 0:
                    prev_tail = chunks[i - 1][-overlap * 4:]
                    chunk = prev_tail + " " + chunk
                overlapped.append(chunk.strip())
            return [c for c in overlapped if c]

    # Fallback: hard split by character
    chars = chunk_size * 4
    return [text[i:i + chars] for i in range(0, len(text), chars - overlap * 4)]


def _clean(text: str) -> str:
    """Normalize whitespace and common OCR artifacts."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ─── Document Processor ──────────────────────────────────────────────────────

class DocumentProcessor:

    def process(self, content: bytes, filename: str, ext: str) -> List[Dict]:
        """
        Returns list of {"text": str, "page": int|None, "sheet": str|None}
        """
        if ext == ".pdf":
            return self._process_pdf(content)
        elif ext in (".xlsx", ".xls"):
            return self._process_excel(content, filename)
        elif ext == ".csv":
            return self._process_csv(content)
        elif ext in (".txt", ".md"):
            return self._process_text(content)
        elif ext == ".docx":
            return self._process_docx(content)
        else:
            raise ValueError(f"Unsupported extension: {ext}")

    # ── PDF ───────────────────────────────────────────────────────────────────

    def _process_pdf(self, content: bytes) -> List[Dict]:
        chunks = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                # Also extract tables
                tables = page.extract_tables()
                for table in tables:
                    rows = ["\t".join(str(c) for c in row if c) for row in table if row]
                    text += "\n\nTable:\n" + "\n".join(rows)
                text = _clean(text)
                for chunk in _split_text(text):
                    chunks.append({"text": chunk, "page": page_num, "sheet": None})
        return chunks

    # ── Excel ─────────────────────────────────────────────────────────────────

    def _process_excel(self, content: bytes, filename: str) -> List[Dict]:
        chunks = []
        xl = pd.ExcelFile(io.BytesIO(content))
        for sheet_name in xl.sheet_names:
            df = xl.parse(sheet_name)
            df = df.dropna(how="all").fillna("")
            # Convert to readable text: header + rows
            text = f"Sheet: {sheet_name}\n"
            text += df.to_string(index=False)
            text = _clean(text)
            for chunk in _split_text(text):
                chunks.append({"text": chunk, "page": None, "sheet": sheet_name})
        return chunks

    # ── CSV ───────────────────────────────────────────────────────────────────

    def _process_csv(self, content: bytes) -> List[Dict]:
        df = pd.read_csv(io.BytesIO(content)).fillna("")
        text = df.to_string(index=False)
        text = _clean(text)
        return [{"text": c, "page": None, "sheet": None} for c in _split_text(text)]

    # ── Text / Markdown ───────────────────────────────────────────────────────

    def _process_text(self, content: bytes) -> List[Dict]:
        text = content.decode("utf-8", errors="ignore")
        text = _clean(text)
        return [{"text": c, "page": None, "sheet": None} for c in _split_text(text)]

    # ── DOCX ─────────────────────────────────────────────────────────────────

    def _process_docx(self, content: bytes) -> List[Dict]:
        doc = DocxDocument(io.BytesIO(content))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n\n".join(paragraphs)
        text = _clean(text)
        return [{"text": c, "page": None, "sheet": None} for c in _split_text(text)]
