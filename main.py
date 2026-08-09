import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader

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

if __name__ == "__main__":

    file_to_load = "sample.txt"
    
    try:
        docs = load_document(file_to_load)
        
        print(f"\nSuccess! Loaded {len(docs)} document(s).")
        print("\n--- Document Sample ---")
        print(docs[0])

        print("\n--- Page Content ---")
        print(docs[0].page_content)
        
        print("\n--- Metadata ---")
        print(docs[0].metadata)
        
    except Exception as e:
        print(f"An error occurred: {e}")