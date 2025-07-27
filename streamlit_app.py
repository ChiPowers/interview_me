import os
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from genai_resume_app.services import vectorstore_service, openai_service
from genai_resume_app.utils.helper_functions import build_prompt

os.environ["faiss_index_path"] = "faiss_index"

st.set_page_config(page_title="Interview Me", layout="centered")
st.title("🧠 Interview Me – Resume Chatbot")
st.markdown("Ask a question about **Chivon Powers's** work history and experience.")

# Define callback handler for streaming
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.text_placeholder = st.empty()
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.text_placeholder.markdown(self.text)

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
            callback_handler = StreamlitCallbackHandler()

            # Pass the callback handler to the streaming function
            answer = openai_service.get_llm_answer_stream(prompt, retriever, question, callback_handler)

            # The answer is streamed token-by-token above, no need to st.success(answer) again here
        except Exception as e:
            st.error(f"Error processing your question: {e}")
