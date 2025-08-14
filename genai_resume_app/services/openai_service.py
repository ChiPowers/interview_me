import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.callbacks.base import BaseCallbackHandler
from langsmith import traceable

from genai_resume_app.services import vectorstore_service
from genai_resume_app.utils.helper_functions import build_prompt

load_dotenv()


class OnChunkCallbackHandler(BaseCallbackHandler):
    """Handles token streaming and ensures clean answer without repeated question."""
    def __init__(self, on_chunk=None):
        self.on_chunk = on_chunk
        self.buffer = []

    def on_llm_new_token(self, token: str, **kwargs):
        cleaned = token.replace("Ġ", " ")  # fix tokenization artifact
        self.buffer.append(cleaned)
        if self.on_chunk:
            self.on_chunk(cleaned)

    def get_final_text(self) -> str:
        return "".join(self.buffer).strip()


def create_qa_chain(memory=None):
    """Factory to create a clean ConversationalRetrievalChain instance."""
    retriever = vectorstore_service.get_most_similar_chunks_for_query("faiss_index")

    llm = ChatOpenAI(
        model_name="gpt-4.1-nano",
        temperature=0.2,
        streaming=False,
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )

    llm_chain = LLMChain(
        llm=llm,
        prompt=build_prompt(),
        input_key="question",
    )

    combine_docs_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        output_key="answer",
    )

    return ConversationalRetrievalChain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain,
        memory=memory,
        question_generator=None,
    )


@traceable(run_type="chain", name="get_answer_auto")
def get_answer_auto(question: str, memory=None, on_chunk=None) -> dict:
    retriever = vectorstore_service.get_most_similar_chunks_for_query("faiss_index")
    handler = OnChunkCallbackHandler(on_chunk)

    llm = ChatOpenAI(
        model_name="gpt-4.1-nano",
        temperature=0.2,
        streaming=bool(on_chunk),
        callbacks=[handler] if on_chunk else [],
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": build_prompt()},
    )

    result = qa_chain.invoke(
        {"question": question},
        config={"return_only_outputs": True},
    )

    final_answer = handler.get_final_text() if on_chunk else result["answer"]
    retrieved_texts = [doc.page_content for doc in retriever.get_relevant_documents(question)]

    return {
        "answer": final_answer,
        "retrieved_context": "\n".join(retrieved_texts),
    }
