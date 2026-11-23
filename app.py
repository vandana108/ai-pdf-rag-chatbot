"""Streamlit UI for the AI PDF Question-Answering Chatbot."""

from __future__ import annotations

import uuid

import streamlit as st
from google import genai

from utils.config import GEMINI_API_KEY, MAX_PDF_SIZE_MB
from utils.rag_pipeline import generate_grounded_answer, index_pdf, retrieve_context

st.set_page_config(
    page_title="AI PDF Chatbot | RAG",
    page_icon="📄",
    layout="centered",
)

st.title("📄 AI PDF Question-Answering Chatbot")
st.write(
    "Upload a PDF and ask questions about its contents. "
    "The assistant retrieves relevant passages and answers with source references."
)

st.info(
    "Privacy notice: Do not upload confidential, sensitive, or personal documents. "
    "Document text is sent to Google's Gemini API to create embeddings and generate answers."
)

if not GEMINI_API_KEY or GEMINI_API_KEY == "PASTE_YOUR_NEW_KEY_HERE":
    st.error(
        "Gemini API key not found. For local use, add it to .env. "
        "For Streamlit Community Cloud, add GEMINI_API_KEY in the app's Secrets settings."
    )
    st.stop()

try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
except Exception:
    st.error("Could not initialize the Gemini client. Check your API configuration.")
    st.stop()

# Each browser session gets a separate Chroma collection namespace.
if "session_namespace" not in st.session_state:
    st.session_state.session_namespace = uuid.uuid4().hex[:12]
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None
if "indexed_filename" not in st.session_state:
    st.session_state.indexed_filename = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.container(border=True):
    st.subheader("1. Upload your PDF")
    uploaded_file = st.file_uploader(
        "Choose a text-based PDF",
        type=["pdf"],
        help=f"Maximum file size: {MAX_PDF_SIZE_MB} MB. Scanned image-only PDFs are not supported in this version.",
    )

    if uploaded_file is not None:
        pdf_bytes = uploaded_file.getvalue()
        st.caption(f"Selected: {uploaded_file.name} · {len(pdf_bytes) / (1024 * 1024):.2f} MB")

        if len(pdf_bytes) > MAX_PDF_SIZE_MB * 1024 * 1024:
            st.error(f"Please upload a PDF smaller than {MAX_PDF_SIZE_MB} MB.")
        elif st.button("Index PDF and start chatting", type="primary", use_container_width=True):
            with st.spinner(
                "Reading the PDF, splitting it into chunks, generating embeddings, and indexing..."
            ):
                try:
                    collection_name, chunk_count, reused = index_pdf(
                        pdf_bytes=pdf_bytes,
                        filename=uploaded_file.name,
                        gemini_client=gemini_client,
                        session_namespace=st.session_state.session_namespace,
                    )
                    st.session_state.collection_name = collection_name
                    st.session_state.indexed_filename = uploaded_file.name
                    st.session_state.messages = []
                    if reused:
                        st.success(
                            f"Your PDF is ready. Reused an existing index with {chunk_count} chunks."
                        )
                    else:
                        st.success(
                            f"PDF indexed successfully. Created {chunk_count} text chunks."
                        )
                except Exception as exc:
                    st.error(f"PDF indexing failed: {exc}")

if st.session_state.collection_name:
    st.success(f"Ready to answer questions about **{st.session_state.indexed_filename}**.")
else:
    st.caption("After selecting a PDF, click the button above to build its searchable index.")

with st.sidebar:
    st.header("Retrieval settings")
    top_k = st.slider(
        "Number of source chunks",
        min_value=1,
        max_value=10,
        value=5,
        help="The number of relevant document chunks passed to the answer-generation step.",
    )
    st.caption("A higher value provides more context but can add irrelevant material.")
    st.divider()
    st.markdown("**How it works**")
    st.markdown("1. Extract PDF text")
    st.markdown("2. Split text into chunks")
    st.markdown("3. Generate embeddings")
    st.markdown("4. Retrieve relevant chunks")
    st.markdown("5. Generate a grounded answer")

st.subheader("2. Ask questions about your document")

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
                        st.caption(f"Cosine distance: {source['distance']:.4f}")

question = st.chat_input(
    "Ask a question about your uploaded PDF...",
    disabled=not bool(st.session_state.collection_name),
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the PDF and preparing an answer..."):
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
                                    f"Cosine distance: {source['distance']:.4f}"
                                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception:
                st.error(
                    "I couldn't complete that request. Check the app logs for details "
                    "or try again in a moment."
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "Sorry, I couldn't complete that request. Please try again.",
                        "sources": [],
                    }
                )

st.divider()
st.caption(
    "Built with Python · Streamlit · Gemini API · Embeddings · ChromaDB · RAG"
)
