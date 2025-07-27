import os
import streamlit as st
from genai_resume_app.services import vectorstore_service, openai_service
from genai_resume_app.utils.helper_functions import build_prompt, session_first_embed_and_store

os.environ["faiss_index_path"] = "faiss_index"
os.environ["rag_pdf_path"] = "docs"

st.set_page_config(page_title="Interview Me", layout="centered")
st.title("🧠 Interview Me – Resume Chatbot")
st.markdown("Ask a question about **Chivon Powers's** work history and experience.")

# Embed documents if not already done
if not os.path.exists(os.environ["faiss_index_path"]):
    with st.spinner("Embedding documents and building vector store for the first time..."):
        session_first_embed_and_store(
            doc_path=os.environ["rag_pdf_path"],
            db_path=os.environ["faiss_index_path"]
        )
    st.success("Embedding complete! You can now ask questions.")

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
            retriever = vectorstore_service.get_most_similar_chunks_for_query(os.environ["faiss_index_path"])
            prompt = build_prompt()
            answer = openai_service.get_llm_answer_stream(prompt, retriever, question, callback_handler=None)  # add your callback handler if streaming
            st.markdown("### 🧠 AI Interview Answer")
            st.success(answer)
        except Exception as e:
            st.error(f"Error processing your question: {e}")
