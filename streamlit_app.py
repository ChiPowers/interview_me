import streamlit as st
from genai_resume_app.services import openai_service

import os
from dotenv import load_dotenv

# Load environment variables early
load_dotenv()

st.set_page_config(page_title="Interview Me", layout="centered")
st.title("🧠 Interview Chivon Powers")
st.markdown(
    "This bot responds as me, using my resume and other documents to answer your interview questions. "
    "It’s a practical demonstration of the AI skills I bring to the table. Ask a question about my work history and experience."
)

# Wrap input in a form for better UX
with st.form(key="qa_form", clear_on_submit=False):
    question = st.text_input(
        label="Enter your interview question:",
        placeholder="e.g., Tell me about a time you solved a tough problem",
        key="user_question"
    )
    submitted = st.form_submit_button("Ask the Question")  # Submit with button or Enter

if submitted:
    if not question:
        st.warning("Please enter a question.")
    else:
        try:
            placeholder = st.empty()
            response_accumulator = [""]  # Mutable container for streaming

            def on_chunk(chunk):
                response_accumulator[0] += chunk
                placeholder.markdown("### 🧠 Answer\n" + response_accumulator[0])

            # Streaming enabled; fallback handled inside openai_service.get_answer_auto
            result = openai_service.get_answer_auto(question, on_chunk=on_chunk)
            if result:
                placeholder.markdown("### 🧠 Answer\n" + result)

        except Exception as e:
            st.error(f"Error processing your question: {e}")
