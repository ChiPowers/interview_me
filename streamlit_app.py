import os
import streamlit as st
from genai_resume_app.services import vectorstore_service, openai_service
from genai_resume_app.utils.helper_functions import build_prompt
from genai_resume_app.utils.custom_callback import SimpleStreamHandler
from dotenv import load_dotenv

load_dotenv() 

os.environ["faiss_index_path"] = "faiss_index"

st.set_page_config(page_title="Interview Me", layout="centered")
st.title("🧠 Interview Me – Resume Chatbot")
st.markdown("Ask questions about my work history and experience.")

question = st.text_input(
    label="Enter your interview question:",
    placeholder="e.g., Tell me about a data science project you built",
    key="user_question"
)

if st.button("Pose the Question"):
    if not question:
        st.warning("Please enter a question.")
    else:
        try:
            retriever = vectorstore_service.get_most_similar_chunks_for_query(os.environ["faiss_index_path"])
            prompt = build_prompt()
            stream_placeholder = st.empty()
            callback_handler = SimpleStreamHandler(stream_placeholder)

            answer = openai_service.get_llm_answer_stream(prompt, retriever, question, callback_handler)

        except Exception as e:
            st.error(f"Error processing your question: {e}")
