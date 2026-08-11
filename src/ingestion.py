import os
import shutil
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = "./data/vectordb"

def load_document(file_path):
    print(f"Loading document: {file_path}")
    
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, "utf-8")
    elif file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    documents = loader.load()
    return documents



def chunk_text(documents): 
    print("\nChunking text...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,      
        chunk_overlap=50,    
        separators=["\n\n", "\n", " ", ""] 
    )

    chunks = text_splitter.split_documents(documents)
    return chunks



def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")



def ingest_document(file_path):
    """
    Main function for Ingestion Pipeline:
    Loads a file, chunks it, generates embeddings, and saves to ChromaDB.
    """
    print(f"\n--- Starting Data Ingestion Pipeline for: {file_path} ---")
    
    docs = load_document(file_path)
    chunks = chunk_text(docs)
    embedding_model = get_embedding_model()
    
    # Clear existing DB if any, and store new chunks
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        
    print(f"Saving {len(chunks)} chunks to ChromaDB...")
    db = Chroma.from_documents(
        chunks, 
        embedding_model, 
        persist_directory=CHROMA_PATH
    )
    print("Ingestion complete! Document successfully stored in ChromaDB.")
    return db