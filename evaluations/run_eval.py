import sys
import os

# Add project root to sys.path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluations.evaluators import conciseness, factual_grounding, relevance, tone, ls_client
from genai_resume_app.services import openai_service
from langchain.memory import ConversationBufferMemory

def chain_factory(example):
    """
    Factory function that receives an example dict and returns a fresh QA chain instance.
    It converts example chat_history into a ConversationBufferMemory for context.
    """
    chat_history = example.get("chat_history", [])
    if chat_history:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        for message in chat_history:
            if message["type"] == "human":
                memory.chat_memory.add_user_message(message["content"])
            elif message["type"] == "ai":
                memory.chat_memory.add_ai_message(message["content"])
    else:
        memory = None

    return openai_service.create_qa_chain(memory=memory)

def main():
    print("\nRunning LangSmith evaluation suite...")

    ls_client.run_on_dataset(
        dataset_name="RAG Application Golden Dataset",
        llm_or_chain_factory=chain_factory,
        evaluators=[conciseness, factual_grounding, relevance, tone, "similarity_score"],
        description="Evaluating interview bot for conciseness, factual grounding, relevance, and tone",
        max_examples=7,
    )
    print("✅ All evaluations complete!")

if __name__ == "__main__":
    main()
