from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from genai_resume_app.utils.helper_functions import format_docs
from langfuse.langchain import CallbackHandler

# Initialize Langfuse callback handler once (reuse this!)
langfuse_handler = CallbackHandler()

def get_llm_answer_with_callback(prompt, retriever, question):
    llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0.2, callbacks=[langfuse_handler])
    
    # Build chain manually as you had it
    rag_chain = (
        {"context": retriever | format_docs, "query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    # Pass question, callbacks already attached to llm
    response = rag_chain.invoke(question)
    return response
