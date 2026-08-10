import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

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


def search_db(query, k=2):
    embedding_model = get_embedding_model()

    db = Chroma(
        persist_directory = "./data/vectordb",
        embedding_function = embedding_model
    )

    results = db.similarity_search(query, k)
    return results


def generate_answer(query, retrieved_chunks):
    """
    Takes the user's query and the chunks we retrieved from the database,
    builds a prompt, and sends it to the Groq LLM to get a final answer.
    """
    print("\nGenerating answer with LLM...")
    
    # Initialize the Groq LLM (it automatically finds the GROQ_API_KEY in the environment)
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    
    context_text = "\n\n".join([chunk.page_content for chunk in retrieved_chunks])
    
    prompt_template = """
    You are a helpful assistant. Answer the question based ONLY on the following context.
    If the answer is not contained in the context, say "I cannot answer this based on the provided documents."
    Do not use any outside knowledge.

    Context:
    {context}

    Question:
    {question}
    """
    
    # Create a LangChain prompt object -> makes an object that takes two inputs and return a prompt
    # useful for chaining operations. we could directly invoke llm with a prompt if we want
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = prompt | llm

    response = chain.invoke({
        "context": context_text,
        "question": query
    })

    return response.content


if __name__ == "__main__":

    file_to_load = "./data/raw/drylab.pdf"
    
    try:
        # PIPELINE A: Ingestion
        # docs = load_document(file_to_load)
        # chunks = chunk_text(docs)
        # embedding_model = get_embedding_model()
        # save_to_chroma(chunks, embedding_model)
        
        # PIPELINE B: Querying
        user_query = "What is the capital of India?"
        retrieved_chunks = search_db(user_query, k=3)
        
        response = generate_answer(user_query, retrieved_chunks)
        print("\n ------- Final Answer --------\n")
        print(response)
            
    except Exception as e:
        print(f"An error occurred: {e}")