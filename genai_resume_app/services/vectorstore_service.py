from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

def load_docs(doc_path):
    """Load PDFs from directory and return list of documents."""
    loader = PyPDFDirectoryLoader(doc_path)
    documents = loader.load()
    return documents

def embed_chunks_and_upload_to_faiss(chunks, index_path):
    """Embed chunks and save FAISS index locally."""
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    faiss_index = FAISS.from_documents(chunks, embeddings)
    faiss_index.save_local(index_path)

def get_most_similar_chunks_for_query(index_path):
    """Load FAISS index and return a retriever for similarity search."""
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    faiss_index = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return retriever
