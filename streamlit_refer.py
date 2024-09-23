import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
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
from matplotlib import font_manager, rc

# 한글 폰트 설정 (나눔고딕)
def set_korean_font():
    # 상대 경로로 폰트 경로 설정 (Streamlit 앱이 있는 폴더에 폰트 파일이 있어야 함)
    font_path = os.path.join(os.getcwd(), "NanumGothic.ttf")
    
    if os.path.exists(font_path):
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        rc('font', family=font_name)
        st.write(f"폰트 {font_name}가 성공적으로 적용되었습니다.")
    else:
        st.error(f"폰트 파일을 찾을 수 없습니다: {font_path}")

# CSV 파일과 이미지 파일 경로 (GitHub에 업로드된 파일을 사용할 경우, 로컬에서 다운로드 받지 않아도 됩니다.)
CSV_PATH = "cardata.csv"  # 상대 경로로 설정
IMAGE_FOLDER_PATH = "images/"  # 이미지 폴더 경로

def main():
    st.set_page_config(
        page_title="CarBot",
        page_icon="\U0001F697"  # 자동차 아이콘
    )

    st.title("_자동차 챗봇 :red[CAR BOT]_ \U0001F697")

    # 한글 폰트 적용
    set_korean_font()

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

    # 차량 선택 및 비교 기능 추가
    st.markdown("## 차량 스펙 비교")
    vehicle1 = st.selectbox("첫 번째 차량을 선택하세요:", st.session_state.car_data['이름'].unique())
    vehicle2 = st.selectbox("두 번째 차량을 선택하세요:", st.session_state.car_data['이름'].unique())

    if st.button("비교하기"):
        compare_vehicles(vehicle1, vehicle2)

    # 이전 대화 출력
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", "content": "안녕하세요! 차량에 대해 궁금하신 것이 있으면 차량의 이름을 입력해주세요!2"}]

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

# CSV 파일 로드 함수
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

# 이미지 파일 로드 함수
def load_vehicle_image(vehicle_number):
    image_path = os.path.join(IMAGE_FOLDER_PATH, f"{vehicle_number}.png")
    if os.path.exists(image_path):
        img = Image.open(image_path)
        return img
    else:
        return None

# 차량 비교 기능
def compare_vehicles(vehicle1, vehicle2):
    vehicle_data = st.session_state.car_data
    vehicle1_data = vehicle_data[vehicle_data['이름'] == vehicle1].iloc[0]
    vehicle2_data = vehicle_data[vehicle_data['이름'] == vehicle2].iloc[0]

    # 차량 이미지 로드
    vehicle1_img = load_vehicle_image(vehicle1_data['차량번호'])
    vehicle2_img = load_vehicle_image(vehicle2_data['차량번호'])

    # 이미지와 차량 이름 출력
    if vehicle1_img and vehicle2_img:
        st.image([vehicle1_img, vehicle2_img], caption=[vehicle1, vehicle2], width=300)
    else:
        st.write("이미지를 불러오지 못했습니다.")

    # 비교할 스펙을 3개의 그룹으로 나누기 (최소가격, 최대가격), (최소연비, 최대연비), (최소출력, 최대출력)
    specs_groups = [
        ('최소가격', '최대가격'),
        ('최소연비', '최대연비'),
        ('최소출력', '최대출력')
    ]

    for specs in specs_groups:
        vehicle1_specs = vehicle1_data[list(specs)].values
        vehicle2_specs = vehicle2_data[list(specs)].values
        
        # 스펙 비교 그래프 그리기
        fig, ax = plt.subplots()
        index = np.arange(len(specs))
        bar_width = 0.35

        bar1 = ax.bar(index, vehicle1_specs, bar_width, label=vehicle1)
        bar2 = ax.bar(index + bar_width, vehicle2_specs, bar_width, label=vehicle2)

        ax.set_xlabel('스펙')
        ax.set_ylabel('값')
        ax.set_title(f'{vehicle1} vs {vehicle2} 스펙 비교 ({specs[0]}, {specs[1]})')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(specs)
        ax.legend()

        # 그래프 출력
        st.pyplot(fig)

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

