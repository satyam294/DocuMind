import json
from src.query_engine import get_answer
from tests.evaluate_rag import evaluate_rag

with open("tests/test_suite.json", "r", encoding="utf-8") as f:
    test_suite = json.load(f)

def run_test_suite():

    print("=" * 70)
    print("RAG EVALUATION SUITE")
    print("=" * 70)

    test_questions = [
       test["question"] for test in test_suite["tests"]
    ]

    all_results = []

    for i, question in enumerate(test_questions, 1):

        print(f"\n{'-' * 70}")
        print(f"TEST {i}")
        print(f"Question: {question}")
        print("-" * 70)

        # Run actual RAG pipeline
        answer, source_chunks = get_answer(question)

        print(f"\nRetrieved chunks: {len(source_chunks)}")
        print(f"Answer: {answer}")

        if not source_chunks:
            print("\nNo context retrieved.")
            continue

        # Evaluate
        results = evaluate_rag(
            question,
            answer,
            source_chunks
        )

        # METRICS
        precision = results["context_precision"]
        sufficiency = results["context_sufficiency"]
        groundedness = results["groundedness"]
        relevance = results["answer_relevance"]

        print("\nMETRICS")
        print("-" * 40)

        print(
            f"Context Precision: "
            f"{precision['score']:.1f}/100"
        )

        print(
            f"Context Sufficiency: "
            f"{sufficiency['score']:.1f}/100 "
            f"(raw: {sufficiency['raw_score']}/4)"
        )

        print(
            f"Answer Groundedness: "
            f"{groundedness['score']:.1f}/100"
        )

        print(
            f"Hallucination Rate: "
            f"{groundedness['hallucination_rate']:.1f}%"
        )

        print(
            f"Answer Relevance: "
            f"{relevance['score']:.1f}/100 "
            f"(raw: {relevance['raw_score']}/4)"
        )

        print(
            f"\nOverall RAG Score: "
            f"{results['overall_score']:.1f}/100"
        )

        all_results.append({
            "question": question,
            "answer": answer,
            **results
        })

    return all_results


if __name__ == "__main__":
    run_test_suite()