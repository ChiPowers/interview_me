
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader


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
	# Set Up Embeddings
	embeddings = OpenAIEmbeddings()
	vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings)
	retriever = vectordb.as_retriever(search_type="similarity_score_threshold",
								    search_kwargs={"score_threshold": 0.5,
						    "k": 3})
	return retriever

