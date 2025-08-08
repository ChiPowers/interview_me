import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.base import BaseCallbackHandler
from genai_resume_app.services import vectorstore_service
from genai_resume_app.utils.helper_functions import build_prompt

load_dotenv()  # ensure .env is loaded early

class OnChunkCallbackHandler(BaseCallbackHandler):
    def __init__(self, on_chunk):
        self.on_chunk = on_chunk

    def on_llm_new_token(self, token: str, **kwargs):
        if self.on_chunk:
            self.on_chunk(token)

# Initialize a global ConversationBufferMemory instance to keep conversation state
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def get_answer_auto(question: str, on_chunk=None) -> str:
    """
    Query the vector store with conversation memory and GPT-4.1-nano.
    Supports streaming via on_chunk callback, otherwise returns full answer.
    """

    retriever = vectorstore_service.get_most_similar_chunks_for_query("faiss_index")
    prompt = build_prompt()

    llm = ChatOpenAI(
        model_name="gpt-4.1-nano",
        temperature=0,
        streaming=bool(on_chunk),
        callbacks=[OnChunkCallbackHandler(on_chunk)] if on_chunk else [],
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )

    # Setup the Conversational Retrieval Chain with memory and your prompt
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    # Run the chain with the question input
    result = qa_chain.invoke({"question": question})

    return result["answer"]
