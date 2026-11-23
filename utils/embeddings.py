"""Gemini embedding helpers."""

from __future__ import annotations

from google import genai
from google.genai import types

from utils.config import EMBEDDING_BATCH_SIZE, GEMINI_EMBEDDING_MODEL


def embed_documents(client: genai.Client, texts: list[str]) -> list[list[float]]:
    """Embed document chunks using the retrieval-document task type."""
    vectors: list[list[float]] = []

    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        response = client.models.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )

        if not response.embeddings or len(response.embeddings) != len(batch):
            raise RuntimeError("Gemini returned an unexpected number of document embeddings.")

        for item in response.embeddings:
            if not item.values:
                raise RuntimeError("Gemini returned an empty document embedding.")
            vectors.append([float(value) for value in item.values])

    return vectors


def embed_query(client: genai.Client, query: str) -> list[float]:
    """Embed a user question using the question-answering task type."""
    response = client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(task_type="QUESTION_ANSWERING"),
    )

    if not response.embeddings or not response.embeddings[0].values:
        raise RuntimeError("Gemini did not return a query embedding.")

    return [float(value) for value in response.embeddings[0].values]
