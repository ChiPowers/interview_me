
from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader

from langchain_community.embeddings import OpenAIEmbeddings
from chromadb.config import Settings

def load_docs(doc_path):
	loader = PyPDFDirectoryLoader(doc_path)
	documents = loader.load()
	return documents


def embed_chunks_and_upload_to_chroma(chunks, db_path):
	# Set Up Embeddings
	embeddings = OpenAIEmbeddings()

	# Store Text in ChromaDBs
	Chroma.from_documents(chunks, embeddings, persist_directory=db_path)
	return


def get_most_similar_chunks_for_query(db_path):
	embedding = OpenAIEmbeddings()
	
	# Set Up Embeddings
	vectordb = Chroma(
    persist_directory="vector_store",
    embedding_function=embedding,
    client_settings=Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="vector_store",  # same directory
    )
)

	retriever = vectordb.as_retriever(search_type="similarity_score_threshold",
								    search_kwargs={"score_threshold": 0.5,
						    "k": 3})
	return retriever

