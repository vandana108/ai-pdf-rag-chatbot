"""Streamlit UI for the AI PDF Question-Answering Chatbot."""

from __future__ import annotations

import streamlit as st
from google import genai

from utils.config import GEMINI_API_KEY, MAX_PDF_SIZE_MB
from utils.rag_pipeline import generate_grounded_answer, index_pdf, retrieve_context

st.set_page_config(
    page_title="AI PDF Chatbot | RAG",
    page_icon="📄",
    layout="wide",
)

st.title("📄 AI PDF Question-Answering Chatbot")
st.caption(
    "Upload a PDF, ask questions, and get answers grounded in the document "
    "with source references."
)

if not GEMINI_API_KEY or GEMINI_API_KEY == "PASTE_YOUR_NEW_KEY_HERE":
    st.error(
        "Gemini API key not found. Copy .env.example to .env, add your API key "
        "to .env, and restart Streamlit."
    )
    st.stop()

try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as exc:
    st.error(f"Could not initialize the Gemini client: {exc}")
    st.stop()

if "collection_name" not in st.session_state:
    st.session_state.collection_name = None
if "indexed_filename" not in st.session_state:
    st.session_state.indexed_filename = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("1. Upload a document")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    st.caption(f"Maximum file size: {MAX_PDF_SIZE_MB} MB")

    if uploaded_file is not None:
        pdf_bytes = uploaded_file.getvalue()
        if len(pdf_bytes) > MAX_PDF_SIZE_MB * 1024 * 1024:
            st.error(f"Please upload a PDF smaller than {MAX_PDF_SIZE_MB} MB.")
        elif st.button("Index PDF", type="primary", use_container_width=True):
            with st.spinner(
                "Extracting text, creating chunks, generating embeddings, and indexing..."
            ):
                try:
                    collection_name, chunk_count, reused = index_pdf(
                        pdf_bytes=pdf_bytes,
                        filename=uploaded_file.name,
                        gemini_client=gemini_client,
                    )
                    st.session_state.collection_name = collection_name
                    st.session_state.indexed_filename = uploaded_file.name
                    st.session_state.messages = []
                    if reused:
                        st.success(
                            f"This PDF is already indexed ({chunk_count} chunks). "
                            "Reusing the saved index."
                        )
                    else:
                        st.success(
                            f"PDF indexed successfully. Created {chunk_count} text chunks."
                        )
                except Exception as exc:
                    st.error(f"Indexing failed: {exc}")

    st.divider()
    st.header("2. Retrieval settings")
    top_k = st.slider(
        "Number of source chunks", min_value=1, max_value=10, value=5
    )
    st.caption("More chunks provide more context, but may add noise to the answer.")

if st.session_state.collection_name:
    st.success(f"Ready to chat with: **{st.session_state.indexed_filename}**")
else:
    st.info(
        "Upload a text-based PDF in the sidebar and click **Index PDF** to begin."
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("View retrieved source references"):
                for source in message["sources"]:
                    st.markdown(
                        f"**{source['filename']} — page {source['page_number']}**  \n"
                        f"{source['text']}"
                    )
                    if source["distance"] is not None:
                        st.caption(f"Retrieval distance: {source['distance']:.4f}")

question = st.chat_input(
    "Ask a question about your uploaded PDF...",
    disabled=not bool(st.session_state.collection_name),
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the document and preparing an answer..."):
            try:
                sources = retrieve_context(
                    collection_name=st.session_state.collection_name,
                    question=question,
                    gemini_client=gemini_client,
                    top_k=top_k,
                )
                answer = generate_grounded_answer(
                    question=question,
                    sources=sources,
                    gemini_client=gemini_client,
                )
                st.markdown(answer)
                if sources:
                    with st.expander("View retrieved source references"):
                        for source in sources:
                            st.markdown(
                                f"**{source['filename']} — page {source['page_number']}**  \n"
                                f"{source['text']}"
                            )
                            if source["distance"] is not None:
                                st.caption(
                                    f"Retrieval distance: {source['distance']:.4f}"
                                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as exc:
                error_message = f"I couldn't complete that request: {exc}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message, "sources": []}
                )

st.divider()
st.caption(
    "Learning project: Python · Gemini API · Embeddings · ChromaDB · RAG · Streamlit"
)
