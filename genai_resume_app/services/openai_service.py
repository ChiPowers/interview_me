import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.base import BaseCallbackHandler
from genai_resume_app.services import vectorstore_service

load_dotenv()  # ensure .env is loaded early

class OnChunkCallbackHandler(BaseCallbackHandler):
    def __init__(self, on_chunk):
        self.on_chunk = on_chunk

    def on_llm_new_token(self, token: str, **kwargs):
        if self.on_chunk:
            self.on_chunk(token)

def get_answer_auto(question: str, on_chunk=None) -> str:
    """
    Query the vector store with conversation memory and GPT-4.1-nano.
    If on_chunk callback is provided, streams tokens as they arrive.
    Otherwise, returns full answer string.
    """
    # Load retriever (FAISS index)
    retriever = vectorstore_service.get_most_similar_chunks_for_query("faiss_index")

    # Setup LLM with or without streaming
    llm = ChatOpenAI(
        model_name="gpt-4.1-nano",
        temperature=0,
        streaming=bool(on_chunk),
        callbacks=[OnChunkCallbackHandler(on_chunk)] if on_chunk else [],
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )

    # Setup conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Build the Conversational Retrieval Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )

    # Run the chain with the question input
    result = qa_chain.invoke({"question": question})

    return result["answer"]
