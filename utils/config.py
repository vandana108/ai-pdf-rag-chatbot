"""Application configuration for local development and Streamlit Community Cloud."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _setting(name: str, default: str = "") -> str:
    """Read a setting from environment variables, then Streamlit secrets."""
    value = os.getenv(name)
    if value:
        return value.strip()

    try:
        secret_value = st.secrets.get(name, default)
    except (FileNotFoundError, KeyError):
        secret_value = default

    return str(secret_value).strip()


GEMINI_API_KEY = _setting("GEMINI_API_KEY")
GEMINI_CHAT_MODEL = _setting("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite")
GEMINI_EMBEDDING_MODEL = _setting(
    "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"
)

_chroma_dir = _setting("CHROMA_DB_DIR", "./chroma_db")
CHROMA_DB_DIR = (PROJECT_ROOT / _chroma_dir).resolve()

MAX_PDF_SIZE_MB = 20
CHUNK_WORDS = 180
CHUNK_OVERLAP_WORDS = 30
EMBEDDING_BATCH_SIZE = 16
DEFAULT_TOP_K = 5
