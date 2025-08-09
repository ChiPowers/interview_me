import streamlit as st
from genai_resume_app.services import openai_service
from genai_resume_app.utils.helper_functions import build_prompt

import os
from dotenv import load_dotenv
import time

# Load environment variables early
load_dotenv()

st.set_page_config(page_title="Interview Me", layout="centered")
st.title("🧠 Interview Chivon Powers")
st.markdown(
    "This bot responds as me, using my resume and other documents to answer your interview questions. "
    "It’s a practical demonstration of the AI skills I bring to the table. Ask a question about my work history and experience."
)

# Initialize memory in session state if not already present
if "memory" not in st.session_state:
    from langchain.memory import ConversationBufferMemory
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Wrap input in a form for better UX
with st.form(key="qa_form", clear_on_submit=False):
    question = st.text_input(
        label="Enter your interview question:",
        placeholder="e.g., Tell me about a time you solved a tough problem",
        key="user_question"
    )
    submitted = st.form_submit_button("Ask the Question")

if submitted:
    if not question or not question.strip():
        st.warning("Please enter a question.")
    else:
        placeholder = st.empty()
        response_accumulator = [""]  # Mutable container for streaming

        clean_question = question.strip().lower()
        skip_echo = [True]  # mutable flag to skip echoing the question text in streaming

        def on_chunk(chunk):
            clean_chunk = chunk.strip().lower()
            # Skip initial chunks that just repeat the question
            if skip_echo[0]:
                if (clean_chunk == clean_question
                    or clean_chunk in clean_question
                    or clean_question in clean_chunk):
                    return
                else:
                    skip_echo[0] = False  # stop skipping chunks now

            time.sleep(0.05)  # slow down streaming for talking pace
            response_accumulator[0] += chunk
            placeholder.markdown("### 🧠 Answer\n" + response_accumulator[0])

        try:
            # Pass memory explicitly to maintain context
            result = openai_service.get_answer_auto(question, st.session_state.memory, on_chunk=on_chunk)
            if result:
                placeholder.markdown("### 🧠 Answer\n" + result)
        except Exception as e:
            st.error(f"Error processing your question: {e}")
