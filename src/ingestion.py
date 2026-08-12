import os
import shutil
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

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
    
    doclist = load_document(file_path)
    embed_and_save(doclist)
    


def ingest_raw_text(raw_text): 
    """
    Bypasses the file loader and ingests raw text directly.
    """
    print("\n--- Starting Data Ingestion Pipeline for raw text ---")
    
    doc = Document(
        page_content=raw_text, 
        metadata={"source": "pasted_text"}
    )
    embed_and_save([doc])
    

    
def embed_and_save(doclist):
    """
    chunking, embedding and saving to ChromaDB
    """
    chunks = chunk_text(doclist)
    embedding_model = get_embedding_model()
        
    print(f"Saving {len(chunks)} chunks to ChromaDB...")
    db = Chroma.from_documents(
        chunks, 
        embedding_model, 
        persist_directory=CHROMA_PATH
    )
    print("Ingestion complete! Document successfully stored in ChromaDB.")
    return db