# DocuMind Project Context

## Project Overview

DocuMind is a lightweight document question-answering application built with Python. It allows users to ingest text and PDF documents, create an embedding-based vector store, and then ask natural language questions against the stored document context. The system is designed to be easy to use and extend, with a Streamlit UI for upload and query interaction.

## Core Functionality

- Upload `.txt` or `.pdf` documents through a Streamlit interface.
- Ingest uploaded documents into a vector database using dense embeddings.
- Store document chunks persistently in ChromaDB.
- Retrieve relevant chunks via semantic similarity search.
- Generate answers to user queries with a Groq-based LLM, constrained to retrieved document context only.

## Architecture

The project is organized into two main pipelines plus a UI layer:

### 1. Ingestion Pipeline (Pipeline A)

Implemented in `src/ingestion.py`:

- `load_document(file_path)`
  - Loads `.txt` files with `TextLoader`.
  - Loads `.pdf` files with `PyPDFLoader`.
- `chunk_text(documents)`
  - Uses `RecursiveCharacterTextSplitter` to split documents into chunks of up to 600 characters with 50 characters overlap.
- `get_embedding_model()`
  - Uses `HuggingFaceEmbeddings` with the model `all-MiniLM-L6-v2`.
- `ingest_document(file_path)`
  - Loads the document, chunks it, computes embeddings, and persists the chunk vectors into ChromaDB at `./data/vectordb`.

This pipeline is responsible for converting raw documents into searchable vector embeddings and storing them for later retrieval.

### 2. Query Pipeline (Pipeline B)

Implemented in `src/query_engine.py`:

- `search_db(query, k=3)`
  - Re-initializes ChromaDB from `./data/vectordb`.
  - Uses the same embedding model to perform similarity search and retrieve the top `k` chunks relevant to the user query.
- `get_answer(query)`
  - Calls `search_db` to retrieve context chunks.
  - Builds a prompt that forces the LLM to answer only from the retrieved context and not use outside knowledge.
  - Uses `ChatGroq` with `llama-3.3-70b-versatile` and `temperature=0` to generate deterministic answers.
  - Returns both the answer text and the retrieved source chunks.

This pipeline is responsible for converting a user question into a context-aware answer backed by the stored documents.

### 3. Streamlit User Interface

Implemented in `app.py`:

- Provides a sidebar for document upload.
- Temporarily saves upload contents to `./data/raw/temp_{filename}` for processing.
- Displays ingestion progress and success status.
- Provides a main panel for question input and answer display.
- Shows retrieved document chunks in an expandable section for traceability.

The UI ties together ingestion and query flows, making the project usable without requiring command-line interaction.

## Folder Structure

```
DocuMind/
├── app.py                   # Streamlit web app entrypoint
├── main.py                  # Simple script to ingest and query a sample file
├── requirements.txt         # Python dependency list
├── PROJECT_CONTEXT.md       # Project context and architecture description
├── .gitignore               # Files and folders excluded from version control
├── .env                     # Environment variables for secrets/configuration
├── data/
│   ├── raw/                 # Raw input documents for ingestion
│   │   ├── chroma.sqlite3   # ChromaDB persistence file (managed by vector db)
│   │   └── <uploaded files>  # Raw documents and temp uploads
│   └── vectordb/            # Persisted embedding database directory
├── src/
│   ├── ingestion.py         # Document loading, chunking, and embedding ingestion
│   └── query_engine.py      # Similarity search and LLM answer generation
└── venv/                    # Local Python virtual environment (ignored by git)
```

## Tools and Technologies Used

- Python
- Streamlit
- LangChain ecosystem libraries:
  - `langchain`
  - `langchain_community`
  - `langchain_huggingface`
  - `langchain_chroma`
  - `langchain_groq`
- Embedding models:
  - `sentence-transformers` via `all-MiniLM-L6-v2`
- Vector database:
  - ChromaDB with persistent storage at `./data/vectordb`
- Document loaders:
  - `TextLoader` for text files
  - `PyPDFLoader` for PDF files
- Environment management:
  - `python-dotenv`
- Additional libraries:
  - `pypdf`

## Data and Storage

- Raw documents are uploaded through the Streamlit UI and temporarily saved in `data/raw/`.
- Ingested document chunks are embedded and persisted in ChromaDB at `data/vectordb`.
- The vector store is reused for semantic search across user queries.
- The `.gitignore` file excludes the virtual environment, `.env`, raw data, and vector DB persistence files.

## Runtime Flow

1. User uploads a document via `app.py`.
2. The document is saved temporarily and passed to `ingest_document()`.
3. The ingestion pipeline loads, chunks, embeds, and stores document data.
4. The user submits a query in the Streamlit UI.
5. `get_answer()` retrieves relevant chunks from ChromaDB.
6. A prompt is constructed and sent to `ChatGroq`.
7. The answer is returned to the UI, along with source chunks for transparency.

## Engineering Insights

- The project cleanly separates ingestion and query logic into two pipelines, which improves maintainability and allows independent testing.
- Reusing the same embedding model for both ingestion and retrieval ensures embedding space consistency.
- Persistent ChromaDB storage makes the application stateful across runs and avoids repeated re-ingestion.
- The prompt explicitly instructs the LLM to answer only from provided context, reducing hallucinations and improving answer reliability.
- The Streamlit UI offers a simple end-user interface while the core logic remains reusable from `main.py`.
- The project is built for extensibility: new loaders, embedding models, similarity search backends, or LLMs can be added without changing the overall structure.
- The existing `main.py` script acts as a lightweight smoke test and demonstration harness for the ingestion and query pipelines.

## Key Project Considerations

- The application currently supports only `.txt` and `.pdf` file types.
- The chunking strategy balances chunk size and overlap to preserve local context while limiting prompt length.
- Answer generation uses a deterministic temperature (`0`) to reduce variability.
- Sensitive configuration data, such as API keys, should be stored in `.env` and not committed to version control.
- The document store path is hardcoded, so deployment should ensure the `data/vectordb` directory is writable.

## Suggested Next Steps

- Add support for additional file formats such as Word documents or HTML.
- Implement better chunk metadata handling to show page numbers or source filenames.
- Add error handling for missing or corrupted ChromaDB persistence data.
- Expand the UI with ingestion history and query logs.
- Add automated tests for ingestion and query pipelines.
