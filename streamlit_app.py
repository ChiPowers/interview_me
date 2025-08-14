import streamlit as st
from PIL import Image
from genai_resume_app.services import openai_service
from genai_resume_app.utils.helper_functions import build_prompt

import os
from dotenv import load_dotenv
import time
import base64
from io import BytesIO


# Load environment variables early
load_dotenv()

# Paths to images
logo_path = "assets/logotat.png"
headshot_path = "assets/cp_face.png"

# Load images safely
try:
    logo = Image.open(logo_path)
except FileNotFoundError:
    logo = None

try:
    headshot = Image.open(headshot_path)
except FileNotFoundError:
    headshot = None

def img_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

logo_b64 = img_to_base64(logo) if logo else ""
headshot_b64 = img_to_base64(headshot) if headshot else ""

st.set_page_config(page_title="Interview Me", layout="centered")

# CSS for styling inputs and buttons + answer layout
st.markdown(
    """
    <style>
    .stTextInput>div>div>input {
        font-size: 18px;
        border-radius: 8px;
        padding: 8px;
        border: 2px solid #4B0082;
    }
    .stButton>button {
        background-color: #4B0082;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 10px 25px;
    }
    .stButton>button:hover {
        background-color: #6A5ACD;
        cursor: pointer;
    }
    /* Center logo and title */
    .header-container {
        text-align: center;
        color: #4B0082;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 18px;
        color: #555555;
        margin-top: 5px;
    }
    /* Answer container with headshot left and text right */
    .answer-container {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        margin-top: 1rem;
    }
    .answer-headshot {
        width: 80px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .answer-text {
        flex-grow: 1;
        font-size: 18px;
        color: #333333;
        white-space: pre-wrap; /* preserves line breaks */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header with logo centered and text below
st.markdown(
    f"""
    <div class="header-container">
        {f'<img src="data:image/png;base64,{logo_b64}" width="150">' if logo_b64 else ""}
        <h1 style="margin-bottom: 0;">Interview Chivon Powers</h1>
        <p class="subtitle">
            This bot responds as me, using my resume and other documents to answer your interview questions.<br>
            It’s a practical demonstration of my AI product development skills and a fun way to learn about my work experience.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize memory in session state if not already present
if "memory" not in st.session_state:
    from langchain.memory import ConversationBufferMemory
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# Wrap input in a form for better UX
with st.form(key="qa_form", clear_on_submit=True):
    question = st.text_input(
        label="Enter your interview question:",
        placeholder="e.g., Tell me about a time you solved a tough problem",
        key="user_question"
    )
    submitted = st.form_submit_button("Ask the Question")

st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

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
            if skip_echo[0]:
                if (clean_chunk == clean_question
                    or clean_chunk in clean_question
                    or clean_question in clean_chunk):
                    return
                else:
                    skip_echo[0] = False

            #time.sleep(0.01)  # slow down streaming for talking pace
            response_accumulator[0] += chunk
            # Render the answer with headshot on the left
            answer_html = f"""
                <div class="answer-container">
                    {f'<img src="data:image/png;base64,{headshot_b64}" class="answer-headshot">' if headshot_b64 else ""}
                    <div class="answer-text">{response_accumulator[0]}</div>
                </div>
            """
            placeholder.markdown(answer_html, unsafe_allow_html=True)

        try:
            result = openai_service.get_answer_auto(question, st.session_state.memory, on_chunk=on_chunk)
            if result:
                # Render final answer again to ensure full text displayed
                final_answer_html = f"""
                    <div class="answer-container">
                        {f'<img src="data:image/png;base64,{headshot_b64}" class="answer-headshot">' if headshot_b64 else ""}
                        <div class="answer-text">{result["answer"]}</div>
                    </div>
                """
                placeholder.markdown(final_answer_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error processing your question: {e}")
