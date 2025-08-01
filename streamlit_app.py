import streamlit as st
from genai_resume_app.services import vectorstore_service, openai_service
from genai_resume_app.utils.helper_functions import build_prompt

INDEX_PATH = "faiss_index"

st.set_page_config(page_title="Interview Me", layout="centered")
st.title("🧠 Interview Me – Resume Chatbot")
st.markdown("Ask a question about **Chivon Powers's** work history and experience.")

# ✅ Wrap input and button in a form
with st.form(key="qa_form", clear_on_submit=False):
    question = st.text_input(
        label="Enter your interview question:",
        placeholder="e.g., Tell me about a time Chivon solved a tough problem",
        key="user_question"
    )
    submitted = st.form_submit_button("Pose the Question")  # Use Enter or button

if submitted:
    if not question:
        st.warning("Please enter a question.")
    else:
        try:
            retriever = vectorstore_service.get_most_similar_chunks_for_query(INDEX_PATH)
            prompt = build_prompt()
            placeholder = st.empty()

            response_accumulator = [""]  # Mutable container for streaming

            def on_chunk(chunk):
                response_accumulator[0] += chunk
                placeholder.markdown("### 🧠 AI Interview Answer\n" + response_accumulator[0])

            # Try streaming first, fallback to sync
            result = openai_service.get_answer_auto(prompt, retriever, question, on_chunk=on_chunk)
            if result:
                placeholder.markdown("### 🧠 AI Interview Answer\n" + result)

        except Exception as e:
            st.error(f"Error processing your question: {e}")
