# 🧠 Interview Me – AI Resume Chatbot

A conversational AI app that lets people ask questions about my work experience. It uses Retrieval-Augmented Generation (RAG) with OpenAI and FAISS to provide accurate, context-rich answers from my resume and CV.

---

## 🔍 Features

- 🔎 Retrieves relevant resume sections using vector search
- 🤖 Uses OpenAI (e.g. GPT-4 or GPT-3.5) to generate intelligent answers
- 💬 Built with Streamlit for a friendly web interface
- 📄 Embeds your resume and CV (PDFs) into ChromaDB
- ⚡ Streams responses for fast and interactive UX

---

## 📁 Project Structure
interview_bot/
│
├── docs/ # Your resume and CV PDFs
├── vector_store/ # ChromaDB directory (auto-generated)
├── genai_resume_app/
│ ├── services/
│ │ ├── chroma_service.py # Retrieves chunks from ChromaDB
│ │ └── openai_service.py # Handles LLM response logic
│ └── utils/
│ └── helper_functions.py # Embedding, PDF loading, prompt builder
│
├── ingest_resume.py # Script to embed and store resume/CV
├── streamlit_app.py # Main Streamlit app
├── requirements.txt # Python dependencies
├── .env # Contains your OPENAI_API_KEY
├── README.md # This file
