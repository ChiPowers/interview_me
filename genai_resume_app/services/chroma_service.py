from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import OpenAIEmbeddings
from chromadb.config import Settings

def load_docs(doc_path):
    loader = PyPDFDirectoryLoader(doc_path)
    documents = loader.load()
    return documents

# Chroma client settings using DuckDB
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="vector_store"
)

def embed_chunks_and_upload_to_chroma(chunks, db_path):
    embeddings = OpenAIEmbeddings()

    Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=db_path,
        client_settings=CHROMA_SETTINGS
    )
    return

def get_most_similar_chunks_for_query(db_path):
    embedding = OpenAIEmbeddings()

    vectordb = Chroma(
        persist_directory=db_path,
        embedding_function=embedding,
        client_settings=CHROMA_SETTINGS
    )

    retriever = vectordb.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.5, "k": 3}
    )
    return retriever
