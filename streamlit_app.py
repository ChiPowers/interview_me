import os
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from genai_resume_app.services import chroma_service, openai_service
from genai_resume_app.utils.helper_functions import build_prompt

# Set paths to pre-embedded data
os.environ["db_path"] = "vector_store"

st.set_page_config(page_title="Interview Me", layout="centered")
st.title("🧠 Interview Me – Resume Chatbot")
st.markdown("Ask a question about **Chivon Powers's** work history and experience.")

question = st.text_input(
    label="Enter your interview question:",
    placeholder="e.g., Tell me about a time Chivon solved a tough problem",
    key="user_question"
)

# Define callback handler here (before use)
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.text = ""
        self.placeholder = st.empty()

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.placeholder.markdown(self.text)

if st.button("Pose the Question"):
    if not question:
        st.warning("Please enter a question.")
    else:
        try:
            retriever = chroma_service.get_most_similar_chunks_for_query(os.environ["db_path"])
            prompt = build_prompt()

            callback_handler = StreamlitCallbackHandler()
            
            # Make sure your openai_service.py has get_llm_answer_stream implemented correctly
            answer = openai_service.get_llm_answer_stream(prompt, retriever, question, callback_handler)
            
            # You can optionally print final answer again or leave streaming output only
            # st.success(answer)

        except Exception as e:
            st.error(f"Error processing your question: {e}")
