import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

judge_llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.0
)


def parse_json_response(response):
    text = response.content.strip()
    
    if text.startswith("```"):
        text = text.replace("```json", "", 1)
        text = text.replace("```", "")
        text = text.strip()
        
    return json.loads(text)


# 1. CONTEXT PRECISION
def grade_context_precision(question, chunks):
    """
    Evaluate every retrieved chunk individually.

    Chunk relevance:
        2 = highly relevant
        1 = partially relevant
        0 = irrelevant

    Final score:
        weighted precision from 0-100
    """

    chunk_text = "\n\n".join(
        f"CHUNK {i + 1}:\n{chunk.page_content}"
        for i, chunk in enumerate(chunks)
    )

    prompt = ChatPromptTemplate.from_template("""
		You are an expert evaluator of a Retrieval-Augmented Generation (RAG) system.

		Your task is to evaluate the PRECISION of the retrieved context.

		Question:
		{question}

		Retrieved chunks:
		{chunks}

		For EVERY chunk, assign a relevance score:

		2 = Highly relevant:
			The chunk contains information directly useful for answering
			the question.

		1 = Partially relevant:
			The chunk is related to the question or provides some useful
			information, but is not directly sufficient.

		0 = Irrelevant:
			The chunk does not provide useful information for answering
			the question.

		Important:
		- Do NOT judge whether the answer itself is correct.
		- Only judge whether each retrieved chunk is useful for answering
		the question.
		- Do not give credit merely because the chunk contains similar
		keywords.
		- Judge semantic usefulness.

		Return ONLY valid JSON in the given format, do not add anything else in the response:
		{{
			"chunks": [
				{{
					"chunk_id": 1,
					"score": 0,
					"reason": "short explanation"
				}}
			]
		}}
	""")

    chain = prompt | judge_llm

    result = chain.invoke({
        "question": question,
        "chunks": chunk_text
    })

    evaluation = parse_json_response(result)

    scores = [
        item["score"]
        for item in evaluation["chunks"]
    ]

    if not scores:
        return {
            "score": 0,
            "chunk_scores": [],
            "relevant_chunks": 0,
            "total_chunks": 0
        }

	# precision = achieved score / total score * 100
    precision = (
        sum(scores) / (2 * len(scores))
    ) * 100

    return {
        "score": round(precision, 2),
        "chunk_scores": scores,
        "relevant_chunks": sum(score > 0 for score in scores),
        "total_chunks": len(scores)
    }



# 2. CONTEXT SUFFICIENCY
def grade_context_sufficiency(question, context):
    """
    Evaluate whether the retrieved context contains enough
    information to answer the question.

    Score:
        4 = completely sufficient
        3 = mostly sufficient
        2 = partially sufficient
        1 = barely useful
        0 = insufficient
    """

    prompt = ChatPromptTemplate.from_template("""
		You are an expert evaluator of a Retrieval-Augmented Generation system.

		Evaluate whether the retrieved context contains enough information
		to answer the question.
        
        First check what information/facts are required to answer the question.
        Then check whether that information is present in the context to assign a rating.
        
        If the question does not explicitly mention any required facts, assign the rating
        based on the details/facts available in the context.

		Question:
		{question}

		Context:
		{context}

		Use this scale:

		4 = Completely sufficient.
			The context contains all important information required to answer.

		3 = Mostly sufficient.
			The context contains the answer but a minor detail is missing.

		2 = Partially sufficient.
			Some important information is present, but substantial information
			required to answer is missing.

		1 = Barely useful.
			The context has weakly related information but cannot reasonably 
            answer the question.

		0 = Completely insufficient.
			The context contains no useful information for answering.

		Return ONLY valid JSON in the given format, do not add anything else in the response:
		{{
			"score": 0,
			"reason": "short explanation"
		}}
	""")

    chain = prompt | judge_llm

    result = chain.invoke({
        "question": question,
        "context": context
    })

    evaluation = parse_json_response(result)

    score = evaluation["score"]

    return {
        "score": round((score / 4) * 100, 2),
        "raw_score": score,
        "reason": evaluation["reason"]
    }



# 3. ANSWER GROUNDEDNESS
def grade_answer_groundedness(context, answer):
    """
    Evaluate every factual claim in the answer.

    Claim score:
        2 = fully supported
        1 = partially supported / ambiguous
        0 = unsupported

    Final groundedness score: 0-100
    """

    prompt = ChatPromptTemplate.from_template("""
		You are an expert evaluator of hallucinations in a RAG system.

		Your task is to evaluate whether every factual claim in the answer
		is supported by the provided context.

		Context:
		{context}

		Answer:
		{answer}

		First break the answer into individual factual claims.

		For each claim assign:

		2 = Fully supported.
			The context directly supports the claim.

		1 = Partially supported or ambiguous.
			The context supports part of the claim, or the wording goes
			slightly beyond what is explicitly supported.

		0 = Unsupported.
			The context provides no evidence for the claim.

		Important:
		- Do not use outside knowledge.
		- A claim is unsupported if it cannot be inferred from the context.
		- Do not penalize harmless wording differences.
		- Focus on factual content.

		Return ONLY valid JSON in the given format, do not add anything else in the response:
		{{
			"claims": [
				{{
					"claim": "atomic factual claim",
					"score": 0,
					"reason": "short explanation"
				}}
			]
		}}
	""")

    chain = prompt | judge_llm

    result = chain.invoke({
        "context": context,
        "answer": answer
    })

    evaluation = parse_json_response(result)

    claims = evaluation["claims"]

    if not claims:
        return {
            "score": 100,
            "hallucination_rate": 0,
            "total_claims": 0,
            "unsupported_claims": 0
        }

    scores = [claim["score"] for claim in claims]

	# percentage of claims backed by the context
    groundedness = (
        sum(scores) / (2 * len(scores))
    ) * 100

    unsupported = sum(score == 0 for score in scores)

	# percentage of claims not backed by the context
    hallucination_rate = (
        unsupported / len(scores)
    ) * 100

    return {
        "score": round(groundedness, 2),
        "hallucination_rate": round(hallucination_rate, 2),
        "total_claims": len(claims),
        "unsupported_claims": unsupported,
        "claims": claims
    }



# 4. ANSWER RELEVANCE
def grade_answer_relevance(question, answer):
    """
    Evaluate whether the answer actually addresses the question.

    Score:
        4 = completely answers
        3 = mostly answers
        2 = partially answers
        1 = barely addresses
        0 = does not answer
    """

    prompt = ChatPromptTemplate.from_template("""
		You are an expert evaluator of RAG answers.

		Evaluate whether the answer actually addresses the user's question.

		Question:
		{question}

		Answer:
		{answer}

		Use this scale:

		4 = Completely relevant.
			Directly answers the question and covers all important aspects.

		3 = Mostly relevant.
			Answers the question but misses a minor aspect.

		2 = Partially relevant.
			Addresses part of the question but leaves an important part
			unanswered.

		1 = Barely relevant.
			Related to the topic but does not meaningfully answer the question.

		0 = Completely irrelevant.
			Does not answer the question.

		Important:
		- Do not judge factual correctness.
		- Do not judge whether the answer is grounded in context.
		- Only judge whether the answer addresses what was asked.
		- Concise answers can still receive a 4 if they fully answer the question.

		Return ONLY valid JSON in the given format, do not add anything else in the response:
		{{
			"score": 0,
			"reason": "short explanation"
		}}
	""")

    chain = prompt | judge_llm

    result = chain.invoke({
        "question": question,
        "answer": answer
    })

    evaluation = parse_json_response(result)

    score = evaluation["score"]

    return {
        "score": round((score / 4) * 100, 2),
        "raw_score": score,
        "reason": evaluation["reason"]
    }



# 5. OVERALL EVALUATION
def evaluate_rag(question, answer, source_chunks):

    context = "\n\n".join(
        chunk.page_content
        for chunk in source_chunks
    )

    if not source_chunks:
        return {
            "context_precision": 0,
            "context_sufficiency": 0,
            "groundedness": 0,
            "answer_relevance": 0
        }

    precision = grade_context_precision(
        question,
        source_chunks
    )

    sufficiency = grade_context_sufficiency(
        question,
        context
    )

    groundedness = grade_answer_groundedness(
        context,
        answer
    )

    relevance = grade_answer_relevance(
        question,
        answer
    )

    overall = (
        0.25 * precision["score"] +
        0.20 * sufficiency["score"] +
        0.35 * groundedness["score"] +
        0.20 * relevance["score"]
    )

    return {
        "context_precision": precision,
        "context_sufficiency": sufficiency,
        "groundedness": groundedness,
        "answer_relevance": relevance,
        "overall_score": round(overall, 2)
    }