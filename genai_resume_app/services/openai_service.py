from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from genai_resume_app.utils.helper_functions import format_docs
from genai_resume_app.utils.custom_callback import SimpleStreamHandler

def get_llm_answer_stream(prompt, retriever, question, callback_handler):
    callback_handler = SimpleStreamHandler()
    llm = ChatOpenAI(
        model_name="gpt-4.1-nano",
        temperature=0.2,
        streaming=True,
        callbacks=[callback_handler]
    )
    rag_chain = (
        {"context": retriever | format_docs, "query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke(question)
