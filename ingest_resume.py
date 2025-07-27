# ingest_resume.py

import os
from dotenv import load_dotenv
from genai_resume_app.utils.helper_functions import session_first_embed_and_store

# ✅ Load environment variables from .env
load_dotenv()

# ✅ Sanity check (you can remove these print lines later)
print("rag_pdf_path:", os.getenv("rag_pdf_path"))
print("db_path:", os.getenv("db_path"))

# ✅ Set them as environment variables for internal calls
os.environ["rag_pdf_path"] = os.getenv("rag_pdf_path")
os.environ["db_path"] = os.getenv("db_path")

session_first_embed_and_store()
print("✅ Resume embedded and stored in vector DB.")
