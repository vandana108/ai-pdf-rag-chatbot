# AI PDF Question-Answering Chatbot (RAG)

A learning-focused portfolio project that lets a user upload a text-based PDF, ask questions about it, and receive answers grounded in retrieved document chunks with source filename and page references.

## Features

- Upload a PDF (maximum 20 MB).
- Extract text page by page with PyPDF.
- Split extracted text into overlapping chunks.
- Generate document embeddings with Gemini using the `RETRIEVAL_DOCUMENT` task type.
- Store text, embeddings, and metadata in ChromaDB.
- Embed user questions using the `QUESTION_ANSWERING` task type.
- Retrieve the nearest document chunks with cosine-distance search.
- Generate an answer grounded in retrieved context with Gemini.
- Display source snippets, filenames, and page numbers.
- Reuse an existing index when the exact same PDF is uploaded again.

## Architecture

1. PDF upload
2. Page-aware text extraction
3. Text chunking with overlap
4. Gemini document embedding generation
5. Persistent ChromaDB indexing
6. Gemini query embedding generation
7. Vector similarity retrieval
8. Grounded LLM answer generation
9. Source-reference display

## Tech stack

- Python
- Streamlit
- Google Gen AI Python SDK
- Gemini API
- `gemini-embedding-001`
- ChromaDB
- PyPDF
- python-dotenv

## Requirements

- Python 3.10 or newer (use a recent ChromaDB release compatible with your Python version).
- A Gemini Developer API key from Google AI Studio.
- Internet connection for Gemini API calls.

## Run locally (Windows PowerShell)

Open PowerShell in the repository root (the folder containing `app.py` and `requirements.txt`).

```powershell
# Create a virtual environment
py -3.14 -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Create your local environment file
Copy-Item .env.example .env
notepad .env
```

In `.env`, replace `PASTE_YOUR_API_KEY_HERE` with your own Gemini API key. Never commit `.env`, paste API keys into source code, or share keys in screenshots or chat.

Start Streamlit:

```powershell
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`. Keep the terminal running while you use the app.

## First test

1. Use a non-confidential, text-based PDF smaller than 20 MB.
2. Upload the PDF in the sidebar.
3. Click **Index PDF** and wait for extraction, chunking, embedding, and indexing to finish.
4. Ask a question whose answer is explicitly present in the PDF.
5. Expand **View retrieved source references** to inspect the source filename, page number, and retrieved text.

## Configuration

The repository includes `.env.example` with safe placeholders. Copy it to `.env` and add your own API key.

The default chat model is `gemini-3.5-flash-lite` and the embedding model is `gemini-embedding-001`. You can change these through environment variables if those model IDs are available to your Gemini API project.

## Current limitations

- Scanned image-only PDFs are not supported; OCR is not included.
- The first version indexes one PDF collection per Streamlit session.
- The local ChromaDB data is stored in `chroma_db/` and is excluded from Git.
- Evaluation metrics, authentication, deployment hardening, and persistent multi-user chat history are future improvements.
- Gemini API usage may be subject to quotas and charges; check current Google AI Studio limits before processing large documents.

## Security

- Keep the real API key in `.env` only.
- `.env`, virtual environments, uploaded PDFs, and local ChromaDB files are excluded by `.gitignore`.
- Use public or sample PDFs for development. Do not upload confidential employer documents or personal records.

## Future improvements

- Add automated RAG evaluation (retrieval quality, answer relevance, and faithfulness).
- Add tracing and observability.
- Add multiple-document collections and persistent chat history.
- Add PDF preview with highlighted source passages.
- Add authentication, rate limits, and secure cloud deployment.

---

Built as a hands-on learning project to understand document processing, embeddings, vector search, and Retrieval-Augmented Generation.
