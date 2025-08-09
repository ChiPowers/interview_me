# ingest_resume.py

import os
from genai_resume_app.services import vectorstore_service 
from genai_resume_app.utils.helper_functions import split_docs, load_docs

def main():
    doc_path = os.environ.get("rag_pdf_path", "docs")
    index_path = os.environ.get("faiss_index_path", "faiss_index")

    print(f"Loading docs from: {doc_path}")
    texts = load_docs(doc_path)
    chunks = split_docs(texts)
    vectorstore_service.embed_chunks_and_upload_to_faiss(chunks, index_path)
    print("Embedding and FAISS index creation complete.")

if __name__ == "__main__":
    main()
