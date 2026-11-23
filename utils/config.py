"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite").strip()
GEMINI_EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"
).strip()

_chroma_dir = os.getenv("CHROMA_DB_DIR", "./chroma_db").strip()
CHROMA_DB_DIR = (PROJECT_ROOT / _chroma_dir).resolve()

MAX_PDF_SIZE_MB = 20
CHUNK_WORDS = 180
CHUNK_OVERLAP_WORDS = 30
EMBEDDING_BATCH_SIZE = 16
DEFAULT_TOP_K = 5
