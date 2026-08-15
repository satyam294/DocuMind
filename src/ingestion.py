import os
import hashlib
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

CHROMA_PATH = "./data/vectordb"

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def get_document_hash(documents):
    full_text = "".join([doc.page_content for doc in documents])
    return hashlib.sha256(full_text.encode('utf-8')).hexdigest()


def is_duplicate(content_hash):
    if not os.path.exists(CHROMA_PATH):
        return False 
        
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_model())
    
    results = db.get(where={"file_hash": content_hash})
    return len(results['ids']) > 0



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



def get_available_documents():
    if not os.path.exists(CHROMA_PATH):
        return []
        
    db = Chroma(persist_directory=CHROMA_PATH)
    
    results = db.get() 
    metadatas = results.get("metadatas", [])
    
    unique_names = set(meta.get("file_name") for meta in metadatas if "file_name" in meta)
    return list(unique_names)



def ingest_document(file_path, original_file_name):
    """
    Main function for Ingestion Pipeline:
    Loads a file, checks for duplicacy, chunks it, generates embeddings, and saves to ChromaDB.
    """
    print(f"\n--- Starting Data Ingestion Pipeline for: {file_path} ---")
    
    doclist = load_document(file_path)

    # Duplicacy Check
    content_hash = get_document_hash(doclist)

    if is_duplicate(content_hash):
        print("Duplicate document found! Skipping ingestion.")
        return False
    
    embed_and_save(doclist, content_hash, original_file_name)
    return True



def ingest_raw_text(raw_text, text_title): 
    """
    Bypasses the file loader and ingests raw text directly.
    """
    print("\n--- Starting Data Ingestion Pipeline for raw text ---")
    
    doc = Document(
        page_content=raw_text, 
        metadata={"source": "pasted_text"}
    )

    # Duplicacy Check
    content_hash = get_document_hash([doc])

    if is_duplicate(content_hash):
        print("Duplicate text found! Skipping ingestion.")
        return False
    
    embed_and_save([doc], content_hash, text_title)
    return True
    

    
def embed_and_save(doclist, content_hash, original_file_name):
    """
    chunking, embedding and saving to ChromaDB
    """
    chunks = chunk_text(doclist)

    # add content_hash to the metadata
    for chunk in chunks:
        chunk.metadata["file_hash"] = content_hash
        chunk.metadata["file_name"] = original_file_name

    embedding_model = get_embedding_model()
        
    db = Chroma.from_documents(
        chunks, 
        embedding_model, 
        persist_directory=CHROMA_PATH
    )
    print("Ingestion complete! Document successfully stored in ChromaDB.")
    return db