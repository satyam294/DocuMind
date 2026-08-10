import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

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
        chunk_size=30,      
        chunk_overlap=5,    
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


if __name__ == "__main__":

    file_to_load = "sample.txt"
    
    try:
        docs = load_document(file_to_load)
        
        print(f"\nSuccess! Loaded {len(docs)} document(s).")

        print("\n--- Page Content ---")
        print(docs[0].page_content)
        
        print("\n--- Metadata ---")
        print(docs[0].metadata)

        chunks = chunk_text(docs)
        print(f"Split into {len(chunks)} chunk(s).")

        embedding_model = get_embedding_model()

        first_chunk_text = chunks[0].page_content
        print(f"\nOriginal Text: '{first_chunk_text[:50]}...'")
        
        # .embed_query() converts a single string of text into a vector
        vector = embedding_model.embed_query(first_chunk_text)
        
        print(f"\nSuccess! The text was converted into a list of {len(vector)} numbers.")
        print("Here are the first 5 numbers of the vector:")
        print(vector[:5])

    except Exception as e:
        print(f"An error occurred: {e}")