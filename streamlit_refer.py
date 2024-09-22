import streamlit as st
import tiktoken
from loguru import logger
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory
import pandas as pd
import os
from io import BytesIO
from PIL import Image

# CSV 파일 경로
CSV_PATH = "cardata.csv"
IMAGE_FOLDER_PATH = "images/"

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

    # 차량 데이터 불러오기
    if "car_data" not in st.session_state:
        st.session_state.car_data = load_vehicle_data()

    if st.session_state.car_data is None:
        st.error("차량 데이터를 불러오지 못했습니다.")
        st.stop()

    # 벡터 스토어 및 대화 체인 생성
    if "conversation" not in st.session_state:
        files_text = st.session_state.car_data.to_string()  # CSV 데이터 전체를 텍스트로 변환
        text_chunks = get_text_chunks(files_text)
        vectorstore = get_vectorstore(text_chunks)
        st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)

    # 이전 대화 출력
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
                with st.spinner("답변을 생성 중입니다..."):
                    try:
                        result = chain({"question": query})
                        response = result['answer']

                        st.markdown(response)

                        # 차량 데이터에서 정보 출력
                        car_info = get_car_info(query)
                        if car_info is not None:
                            st.markdown("### 차량 정보")
                            st.dataframe(car_info)
                            photo_number = car_info['차량번호'].values[0]
                            image_path = os.path.join(IMAGE_FOLDER_PATH, f"{photo_number}.png")
                            logger.info(f"Looking for image at: {image_path}")
                            display_vehicle_image(image_path)

                        with st.expander("참고 문서 확인"):
                            for doc in result.get('source_documents', []):
                                st.markdown(f"{doc.metadata['source']}", help=doc.page_content)

                        st.session_state.messages.append({"role": "assistant", "content": response})

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            else:
                st.error("Conversation chain is not initialized. Please process the documents first.")

# CSV 파일 로드
def load_vehicle_data():
    try:
        # 파일 경로 및 존재 여부 확인
        if os.path.exists(CSV_PATH):
            st.write(f"파일 경로 확인됨: {CSV_PATH}")
            data = pd.read_csv(CSV_PATH)
            st.success("차량 데이터를 성공적으로 불러왔습니다.")
            return data
        else:
            st.error(f"CSV 파일이 경로에 존재하지 않습니다: {CSV_PATH}")
            return None
    except Exception as e:
        st.error(f"차량 데이터를 불러오는 중 오류가 발생했습니다: {e}")
        logger.error(f"차량 데이터를 불러오는 중 오류가 발생했습니다: {e}")
        return None

# 이미지 파일 표시
def display_vehicle_image(image_path):
    if os.path.exists(image_path):
        img = Image.open(image_path)
        st.image(img, caption=f"차량 이미지")
    else:
        st.error("차량 이미지를 불러올 수 없습니다.")

# 텍스트를 청크로 나누기
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    return text_splitter.split_text(text)

# 벡터 스토어 생성
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return FAISS.from_texts(text_chunks, embeddings)

# 대화 체인 생성
def get_conversation_chain(vetorestore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=1, max_tokens=500)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vetorestore.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        return_source_documents=True,
        verbose=True
    )

# 차량 정보 검색
def get_car_info(query):
    if st.session_state.car_data is not None:
        df = st.session_state.car_data
        result = df[df.apply(lambda row: query.lower() in row.to_string().lower(), axis=1)]
        if not result.empty:
            return result
    return None

# 텍스트 토큰 길이 계산
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(text))

if __name__ == '__main__':
    main()
