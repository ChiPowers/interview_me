from langchain_core.callbacks.base import BaseCallbackHandler
import streamlit as st

class SimpleStreamHandler(BaseCallbackHandler):
    def __init__(self, *args, **kwargs):
        self.output_container = st.empty()
        self.accumulated_text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.accumulated_text += token
        self.output_container.markdown(self.accumulated_text + "▌")
