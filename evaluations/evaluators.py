from langsmith import Client
from langsmith.evaluation import EvaluationResult
import os

ls_client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))

def conciseness(run, example):
    answer = run.outputs.get("answer", "")
    score = max(1, min(5, 5 - (len(answer.split()) // 40)))  # Shorter → higher score
    return EvaluationResult(key="conciseness", score=score, comment="Shorter answers score higher")

def factual_grounding(run, example):
    # This is a placeholder — replace with retrieval-based check if needed
    score = 4 if example.inputs["question"].lower() in run.outputs.get("answer", "").lower() else 2
    return EvaluationResult(key="factual_grounding", score=score, comment="Checks if answer mentions the question topic")

def relevance(run, example):
    question = example.inputs["question"].lower()
    answer = run.outputs.get("answer", "").lower()
    score = 5 if any(word in answer for word in question.split()) else 2
    return EvaluationResult(key="relevance", score=score, comment="Higher score if key terms overlap")

def tone(run, example):
    answer = run.outputs.get("answer", "").lower()
    score = 5 if any(word in answer for word in ["professional", "enthusiastic", "collaborative"]) else 3
    return EvaluationResult(key="tone", score=score, comment="Looks for positive, professional tone words")
