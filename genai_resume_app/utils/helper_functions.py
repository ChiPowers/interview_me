# YourApp/utils/helper_functions.py

from genai_resume_app.services import vectorstore_service
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader

import os

def load_docs(path_to_pdfs):
    loader = PyPDFDirectoryLoader(path_to_pdfs)
    documents = loader.load()
    return documents


def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    # Split Text into Manageable Chunks
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False)
    texts = text_splitter.split_documents(documents)
    return texts


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_prompt():
    from langchain_core.prompts import ChatPromptTemplate
    template = """
    You are interviewing for an Applied AI scientist position at a tech company.\
        Use the following context to answer interview questions in a way that describes how your experience \
        and skills relate to the job requirements. Use no more than 2 short sentences.\

        Context: {context} \
        Interview Question: \
        {query} \
        Answer:
        """

    prompt = ChatPromptTemplate.from_template(template)
    return prompt

def session_first_embed_and_store(doc_path=None, db_path=None):
    if doc_path is None:
        doc_path = os.environ.get("rag_pdf_path")
    if db_path is None:
        db_path = os.environ.get("db_path")

    if not doc_path or not db_path:
        raise ValueError("Both doc_path and db_path must be set.")

    texts = load_docs(doc_path)
    chunks = split_docs(texts)
    vectorstore_service.embed_chunks_and_upload_to_chroma(chunks, db_path)
    return
 
