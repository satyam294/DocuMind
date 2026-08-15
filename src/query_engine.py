from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

CHROMA_PATH = "./data/vectordb"

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def search_db(query, k=3, selected_files=None):
    embedding_model = get_embedding_model()

    db = Chroma(
        persist_directory = CHROMA_PATH,
        embedding_function = embedding_model
    )

    if selected_files:
        if len(selected_files) == 1:
            filter_dict = {"file_name": selected_files[0]}
        else:
            filter_dict = {"file_name": {"$in": selected_files}}

        return db.similarity_search(query, k=k, filter=filter_dict)
            
    return db.similarity_search(query, k=k)
    


def get_answer(query, selected_files=None):
    """
    Main function of Answer generation pipeline:
    Takes the user's query and retrieves chunks from the database,
    builds a prompt, and sends it to the Groq LLM to get a final answer.
    """

    print(f"\n--- Starting Query Pipeline for: '{query}' ---")

    retrieved_chunks = search_db(query, selected_files=selected_files)

    if not retrieved_chunks:
        return "No relevant context found in the selected documents.", []
    
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

    # Initialize the Groq LLM (it automatically finds the GROQ_API_KEY in the environment)
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

    chain = prompt | llm

    response = chain.invoke({
        "context": context_text,
        "question": query
    })

    return response.content, retrieved_chunks