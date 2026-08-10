import os
import shutil
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = "./data/vectordb"

def load_document(file_path):
    """
    Takes a file path, determines the file extension, and uses the 
    appropriate LangChain document loader to extract the text.
    """
    print(f"Loading document: {file_path}")
    
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, "utf-8")
    elif file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    # list of Document objects
    documents = loader.load()
    return documents



def chunk_text(documents): 
    """
    Takes a list of LangChain Documents and splits them into smaller chunks.
    """
    print("\nChunking text...")
    
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,      
        chunk_overlap=50,    
        separators=["\n\n", "\n", " ", ""] 
    )
    
    # Split the documents
    chunks = text_splitter.split_documents(documents)
    return chunks


def get_embedding_model(): # NEW FUNCTION
    """
    Initializes and returns our embedding model.
    """
    print("Initializing embedding model...")
    model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return model


def save_to_chroma(chunks, embedding_model): 
    """
    Takes the text chunks and the embedding model, converts the chunks to vectors,
    and saves them, with original chunk in a local Chroma database.
    """
    # Clear out the old database folder if it exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        
    print(f"Saving {len(chunks)} chunks to ChromaDB...")
    
    # Create a new Chroma database from our documents
    # LangChain implicitly runs the embedding model on the chunks
    db = Chroma.from_documents(
        chunks, 
        embedding_model, 
        persist_directory=CHROMA_PATH
    )
    
    print(f"Successfully saved to {CHROMA_PATH} directory!")
    return db


if __name__ == "__main__":

    file_to_load = "./data/raw/drylab.pdf"
    
    try:
        docs = load_document(file_to_load)
        print(f"\nSuccess! Loaded {len(docs)} document(s).")

        chunks = chunk_text(docs)
        print(f"Split into {len(chunks)} chunk(s).")

        embedding_model = get_embedding_model()
        
        # Save them to the database -> lang_chroma runs the embedding model and stores vectors
        db = save_to_chroma(chunks, embedding_model)

    except Exception as e:
        print(f"An error occurred: {e}")