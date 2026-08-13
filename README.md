# DocuMind

A lightweight document question-answering system that lets you upload documents and ask natural language questions. Built with Python, LangChain, ChromaDB, and Groq LLM.

## Architecture

```
Documents (.txt/.pdf)
        ↓
[Ingestion Pipeline]
  • Load document
  • Split into chunks (RecursiveCharacterTextSplitter)
  • Generate embeddings (HuggingFace all-MiniLM-L6-v2)
  • Store in ChromaDB
        ↓
    Vector Store (ChromaDB)
        ↓
[Query Pipeline]
  • Semantic search → retrieve top-k chunks
  • Build context + prompt
  • LLM answer (Groq llama-3.3-70b)
        ↓
    User Answer + Source Chunks
```

## Key Features

- **Multi-format support**: Upload `.txt` or `.pdf` documents, or paste raw text
- **Semantic search**: Uses dense embeddings to find relevant document chunks
- **Persistent storage**: ChromaDB vector database stored locally
- **LLM-powered answers**: Groq-based LLM constrained to document context only
- **Streamlit UI**: Easy-to-use web interface for upload and query

## Project Structure

```
DocuMind/
├── app.py                 # Streamlit web UI (main entry point)
├── main.py               # CLI-based ingestion and query example
├── requirements.txt      # Python dependencies
├── src/
│   ├── ingestion.py      # Document loading, chunking, embedding
│   └── query_engine.py   # Semantic search and LLM answer generation
└── data/
    ├── raw/              # Uploaded documents
    └── vectordb/         # ChromaDB persistent storage
```

## Installation

1. **Clone/navigate to the project**:
   ```bash
   cd DocuMind
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (create `.env` file):
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Running the Project

### Option 1: Web UI (Recommended)
```bash
streamlit run app.py
```
- Upload or paste documents in the sidebar
- Ask questions in the main panel
- View retrieved chunks for traceability

### Option 2: CLI Script
```bash
python main.py
```
- Ingests `./data/raw/drylab.pdf` (edit the path as needed)
- Runs sample questions
- Prints answers to console

## How It Works

1. **Ingestion**: Documents are split into 600-char chunks (50-char overlap), embedded using HuggingFace, and stored in ChromaDB
2. **Query**: User questions are embedded and compared against stored chunks; top-3 similar chunks are retrieved
3. **Answer**: Retrieved chunks are passed to Groq LLM with a system prompt that constrains answers to document context only

## Dependencies

- `langchain` + `langchain-community` + ecosystem packages
- `chromadb` - Vector database
- `sentence-transformers` - Embeddings
- `langchain-groq` - LLM integration
- `streamlit` - Web UI
- `pypdf` - PDF parsing

## Environment Variables

- `GROQ_API_KEY` - API key for Groq LLM access (required)

## Notes

- Answers are generated only from document context; out-of-context questions will be refused
- ChromaDB persists vectors locally for fast retrieval without re-embedding
- Temperature is set to 0 for deterministic answers
