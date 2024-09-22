import streamlit as st
import tiktoken
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

import pandas as pd
import os
import requests
from io import BytesIO
from PIL import Image

# GitHub에 업로드된 파일의 raw URL
CSV_URL = "https://raw.githubusercontent.com/wodud2565/langchain/main/cardata.csv"
IMAGE_BASE_URL = "https://raw.githubusercontent.com/wodud2565/langchain/main/images/"

def main():
    st.set_page_config(
        page_title="CarBot",
        page_icon="\U0001F697"  # 자동차 아이콘
    )

    st.title("_자동차 챗봇 :red[CAR BOT]_ \U0001F697")

    # OpenAI API 키 입력란 추가
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

    if not openai_api_key:
        st.sidebar.warning("Please provide your OpenAI API key.")
        st.stop()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    if "car_data" not in st.session_state:
        st.session_state.car_data = None

    # GitHub에서 CSV 파일 로드
    if "car_data" not in st.session_state:
        st.session_state.car_data = load_vehicle_data()

    if st.session_state.car_data is None:
        st.error("차량 데이터를 불러오지 못했습니다.")
        st.stop()

    if "conversation" not in st.session_state:
        # 벡터 스토어를 만들고 대화 체인 생성
        files_text = get_text_from_csv(st.session_state.car_data)
        text_chunks = get_text_chunks(files_text)
        vectorstore = get_vectorstore(text_chunks)
        st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", "content": "안녕하세요! 차량에 대해 궁금하신 것이 있으면 차량의 이름을 입력해주세요!"}]

    # 대화 기록 출력
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 질문 입력
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

                        # 차량 데이터에서 정보 출력
                        if st.session_state.car_data is not None:
                            car_info = get_car_info(query)
                            if car_info is not None:
                                st.markdown("### 차량 정보")
                                st.dataframe(car_info)
                                photo_number = car_info['차량번호'].values[0]
                                image_path = f"{IMAGE_BASE_URL}{photo_number}.png"
                                logger.info(f"Looking for image at: {image_path}")
                                display_vehicle_image(image_path)

                        with st.expander("참고 문서 확인"):
                            for doc in source_documents:
                                st.markdown(f"{doc.metadata['source']}", help=doc.page_content)

                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            else:
                st.error("Conversation chain is not initialized. Please process the documents first.")

# GitHub에서 차량 데이터 CSV 불러오기
def load_vehicle_data():
    try:
        return pd.read_csv(CSV_URL)
    except Exception as e:
        logger.error(f"Error loading vehicle data: {e}")
        return None

# 차량 이미지 표시
def display_vehicle_image(image_url):
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            st.image(img, caption=f"차량 이미지")
        else:
            st.error("차량 이미지를 불러올 수 없습니다.")
    except Exception as e:
        st.error(f"Error loading image: {e}")

# CSV 파일에서 텍
