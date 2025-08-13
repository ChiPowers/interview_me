import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.base import BaseCallbackHandler
from genai_resume_app.services import vectorstore_service
from genai_resume_app.utils.helper_functions import build_prompt

load_dotenv()
def create_qa_chain(memory=None):
    """
    Factory function to create a ConversationalRetrievalChain instance.
    Pass `memory` if you want to keep conversation context.
    """
    retriever = vectorstore_service.get_most_similar_chunks_for_query("faiss_index")
    prompt = build_prompt()

    llm = ChatOpenAI(
        model_name="gpt-4.1-nano",
        temperature=0.2,
        streaming=False,  # No streaming for evaluations
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    return qa_chain


class OnChunkCallbackHandler(BaseCallbackHandler):
    def __init__(self, on_chunk):
        self.on_chunk = on_chunk

    def on_llm_new_token(self, token: str, **kwargs):
        if self.on_chunk:
            self.on_chunk(token)

def get_answer_auto(question: str, memory, on_chunk=None) -> str:
    """
    Synchronous or streaming LLM call with conversation memory and retriever.
    Pass on_chunk callback to stream tokens progressively.
    """
    retriever = vectorstore_service.get_most_similar_chunks_for_query("faiss_index")
    prompt = build_prompt()

    llm = ChatOpenAI(
        model_name="gpt-4.1-nano",
        temperature=0.2,
        streaming=bool(on_chunk),
        callbacks=[OnChunkCallbackHandler(on_chunk)] if on_chunk else [],
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    # Run the chain
    result = qa_chain.invoke({"question": question}, config={"metadata": {"step": "pass_q_thru_chain"}})

    # Get retrieved documents for logging
    retrieved_docs = retriever.get_relevant_documents(question)
    retrieved_texts = [doc.page_content for doc in retrieved_docs]

    # Log to LangSmith so evaluators can use this
    from langsmith import client as ls_client
    ls_client.log_outputs({
        "answer": result["answer"],
        "retrieved_context": "\n".join(retrieved_texts)
    })

    return result["answer"]
