import os
import streamlit as st
from genai_resume_app.services import vectorstore_service, openai_service
from genai_resume_app.utils.helper_functions import build_prompt

INDEX_PATH = "faiss_index"

st.set_page_config(page_title="Interview Me", layout="centered")
st.title("🧠 Interview Me – Resume Chatbot")
st.markdown("Ask a question about **Chivon Powers's** work history and experience.")

question = st.text_input(
    label="Enter your interview question:",
    placeholder="e.g., Tell me about a time Chivon solved a tough problem",
    key="user_question"
)

if st.button("Pose the Question"):
    if not question:
        st.warning("Please enter a question.")
    else:
        try:
            retriever = vectorstore_service.get_most_similar_chunks_for_query(INDEX_PATH)
            prompt = build_prompt()
            
            # Call your updated function with Langfuse callback tracing baked in
            answer = openai_service.get_llm_answer_with_callback(prompt, retriever, question)

            st.markdown("### 🧠 AI Interview Answer")
            st.success(answer)
        except Exception as e:
            st.error(f"Error processing your question: {e}")
