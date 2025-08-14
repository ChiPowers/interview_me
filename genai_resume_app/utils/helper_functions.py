# YourApp/utils/helper_functions.py

from genai_resume_app.services import vectorstore_service
from langchain_community.document_loaders import PyPDFDirectoryLoader
import os
from langchain.docstore.document import Document
from langchain.text_splitter import (
    TextSplitter
)

from dataclasses import dataclass
# from langchain.schema import Document

from langchain.text_splitter import TextSplitter
from langchain.schema import Document

class AcademicCVSplitter(TextSplitter):
    """Simple PDF text splitter that works on Document objects."""
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, doc: Document) -> list[Document]:
        """Split a Document into chunks, each as a Document."""
        text = doc.page_content
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks = []
        for para in paragraphs:
            if len(para) <= self.chunk_size:
                chunks.append(Document(page_content=para))
            else:
                for i in range(0, len(para), self.chunk_size - self.chunk_overlap):
                    chunk_text = para[i:i + self.chunk_size]
                    chunks.append(Document(page_content=chunk_text))
        return chunks

def load_docs(path_to_pdfs):
    loader = PyPDFDirectoryLoader(path_to_pdfs)
    documents = loader.load()
    return documents

def split_docs(docs):
    splitter = AcademicCVSplitter()
    chunks = []
    for doc in docs:
        chunks.extend(splitter.split_text(doc))
    return chunks

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

from langchain_core.prompts import ChatPromptTemplate

def build_prompt():
    template = """
You are Chivon E. Powers interviewing for data science roles. Use the context to answer questions conversationally.

Context: {context}
Question: {question}
Answer:"""
    return ChatPromptTemplate.from_template(template)

def session_first_embed_and_store(doc_path=os.environ.get("rag_pdf_path"),
                                  db_path=os.environ.get("faiss_index_path")):
    texts = load_docs(doc_path)
    chunks = split_docs(texts)
    vectorstore_service.embed_chunks_and_upload_to_faiss(chunks, db_path)

 
