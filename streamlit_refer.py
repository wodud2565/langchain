import streamlit as st
import tiktoken
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader, CSVLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

import pandas as pd
import os

def main():
    st.set_page_config(
        page_title="DirChat",
        page_icon="\U0001F697"  # 자동차 아이콘
    )

    st.title("_Private Data :red[QA Chat]_ \U0001F697")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    if "car_data" not in st.session_state:
        st.session_state.car_data = None

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx', 'csv'], accept_multiple_files=True)
        uploaded_images = st.file_uploader("Upload images", type=['png'], accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")

    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)

        st.session_state.conversation = get_conversation_chain(vetorestore, openai_api_key)

        if uploaded_files:
            for file in uploaded_files:
                if file.name.endswith('.csv'):
                    st.session_state.car_data = pd.read_csv(file)

        save_uploaded_images(uploaded_images)
        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant",
                                         "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            if st.session_state.conversation:
                chain = st.session_state.conversation

                with st.spinner("Thinking..."):
                    try:
                        result = chain({"question": query})
                        with get_openai_callback() as cb:
                            st.session_state.chat_history = result['chat_history']
                        response = result['answer']
                        source_documents = result['source_documents']

                        st.markdown(response)
                        if st.session_state.car_data is not None:
                            c
