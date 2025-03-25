import streamlit as st
import requests
import os
from typing import Dict, Optional
from dotenv import load_dotenv
import logging
from datetime import datetime

load_dotenv()
logging.basicConfig(level="INFO", format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

API_URL = os.getenv("API_URL")
if not API_URL:
    logger.error("API_URL is not set in environment variables.")
    raise ValueError("API_URL must be set to the deployed FastAPI backend URL.")

PROMPT_VERSIONS = ["1.0", "1.1"]
MAX_FILE_SIZE_MB = 5

def setup_page():
    st.set_page_config(page_title="AI Knowledge Assistant", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

def query_api(query: str, file: Optional[st.runtime.uploaded_file_manager.UploadedFile] = None, 
              prompt_version: str = "1.1", history: list = []) -> Dict:
    try:
        if file:
            file_size_mb = len(file.read()) / (1024 * 1024)
            file.seek(0)
            if file_size_mb > MAX_FILE_SIZE_MB:
                raise ValueError(f"File size ({file_size_mb:.2f} MB) exceeds {MAX_FILE_SIZE_MB} MB limit.")
            files = {"file": (file.name, file.read(), file.type)}
            data = {"query": query, "prompt_version": prompt_version, "history": str(history)}
            response = requests.post(f"{API_URL}/query_with_file", data=data, files=files, timeout=30)
        else:
            payload = {"query": query, "prompt_version": prompt_version, "history": str(history)}
            response = requests.post(f"{API_URL}/query", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"API request failed: {str(e)}")
        raise

def main():
    try:
        setup_page()
        st.sidebar.title("AI Knowledge Assistant for Startups")
        st.sidebar.markdown("Chat with an AI to query or summarize documents (max 5 MB).")
        st.sidebar.markdown("You can use this document to upload:- https://drive.google.com/file/d/11l5rKEX-TjRgaGJes2Y_crPx7TlwybSd/view?usp=sharing.")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "file_uploaded" not in st.session_state:
            st.session_state.file_uploaded = None

        st.title("AI Knowledge Assistant Chatbot")
        uploaded_file = st.file_uploader(f"Upload a document (PDF) - Max {MAX_FILE_SIZE_MB} MB", 
                                        type=["pdf"])
        
        if uploaded_file and uploaded_file != st.session_state.file_uploaded:
            st.session_state.file_uploaded = uploaded_file
            result = query_api("load file", uploaded_file, "1.1")
            st.session_state.chat_history.append({"role": "system", "content": result["response"], 
                                                  "timestamp": datetime.utcnow().isoformat() + "Z"})

        prompt_version = st.sidebar.selectbox("Prompt Version", PROMPT_VERSIONS, index=1)
        
        with st.form(key="chat_form", clear_on_submit=True):
            query = st.text_input("You:", placeholder="Ask a question or request a summary...")
            col1, col2 = st.columns([1, 1])
            with col1:
                submit_button = st.form_submit_button("Send")
            with col2:
                clear_button = st.form_submit_button("Clear Chat")

        if clear_button:
            st.session_state.chat_history = []
            st.session_state.file_uploaded = None
            st.experimental_rerun()

        if submit_button and query.strip():
            with st.spinner("Processing..."):
                result = query_api(query, st.session_state.file_uploaded, prompt_version, 
                                  [msg["content"] for msg in st.session_state.chat_history])
                st.session_state.chat_history.append({"role": "user", "content": query, 
                                                      "timestamp": datetime.utcnow().isoformat() + "Z"})
                st.session_state.chat_history.append({"role": "assistant", "content": result["response"], 
                                                      "timestamp": datetime.utcnow().isoformat() + "Z"})

        st.subheader("Conversation")
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"**You ({msg['timestamp']}):** {msg['content']}")
                elif msg["role"] == "assistant":
                    st.markdown(f"**Assistant ({msg['timestamp']}):** {msg['content']}")
                else:
                    st.markdown(f"*{msg['content']} ({msg['timestamp']})*")

        st.markdown("---")
        st.markdown(f"Developer by Yash Narkhede email:- yashnarkhede03@gmail.com | API: {API_URL} | Date: 2025-03-25")
    except Exception as e:
        st.error(f"Error starting the application: {str(e)}")
        logger.error(f"Application startup failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    logger.info("Starting Streamlit app")
    main()