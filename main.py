from src.ingestion import ingest_document
from src.query_engine import get_answer

if __name__ == "__main__":
    file_path = "./data/raw/drylab.pdf"
    
    try:
        # Pipeline A: Ingest document
        ingest_document(file_path)
        
        # Pipeline B: Ask questions
        question_1 = "Summarize sales data in 3 points."
        answer_1, _ = get_answer(question_1)
        
        print("\n--- ANSWER ---")
        print(answer_1)
        
        # Test out-of-context query
        question_2 = "Who is Napoleon?"
        answer_2, _ = get_answer(question_2)
        
        print("\n--- ANSWER ---")
        print(answer_2)
        
    except Exception as e:
        print(f"An error occurred: {e}")