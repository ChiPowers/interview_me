import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from genai_resume_app.utils.helper_functions import format_docs
from langfuse.langchain import CallbackHandler

# Initialize Langfuse callback handler
langfuse_handler = CallbackHandler()


def get_llm_answer(prompt, retriever, question):
    """
    Synchronous version of the LLM call (non-streaming).
    Used as a fallback if async isn't available.
    """
    llm = ChatOpenAI(
        model_name="gpt-4.1-nano",
        temperature=0.2,
        callbacks=[langfuse_handler]
    )
    rag_chain = (
        {"context": retriever | format_docs, "query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    response = rag_chain.invoke(question)
    return response


async def get_llm_answer_with_stream(prompt, retriever, question):
    """
    Async streaming version of the LLM call.
    Yields chunks of the response as they're generated.
    """
    llm = ChatOpenAI(
        model_name="gpt-4.1-nano",
        temperature=0.2,
        streaming=True,
        callbacks=[langfuse_handler]
    )
    rag_chain = (
        {"context": retriever | format_docs, "query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    async for chunk in rag_chain.astream(question):
        yield chunk


def get_answer_auto(prompt, retriever, question, on_chunk=None):
    """
    Auto-detect whether async streaming can be used.
    - If streaming works: returns chunks via on_chunk callback
    - Otherwise: falls back to normal synchronous call
    """
    try:
        async def run_stream():
            async for chunk in get_llm_answer_with_stream(prompt, retriever, question):
                if on_chunk:
                    on_chunk(chunk)

        asyncio.run(run_stream())
        return None  # streaming handled via callback
    except RuntimeError:
        # Fallback to non-async version (e.g., in environments that don't support asyncio.run)
        return get_llm_answer(prompt, retriever, question)
