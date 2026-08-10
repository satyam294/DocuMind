import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

        print("\n--- First Chunk ---")
        print(chunks[0].page_content)
        
        if len(chunks) > 1:
            print("\n--- Second Chunk ---")
            print(chunks[1].page_content)
        
    except Exception as e:
        print(f"An error occurred: {e}")