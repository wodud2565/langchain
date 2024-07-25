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
from tempfile import NamedTemporaryFile

def main():
    st.set_page_config(
        page_title="DirChat",
        page_icon=":books:"
    )

    st.title("_Private Data :red[QA Chat]_ :books:")

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
                    if 'color_code' in st.session_state.car_data.columns:
                        st.session_state.car_data = st.session_state.car_data.drop(columns=['color_code'])

        save_uploaded_images(uploaded_images)
        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant",
                                         "content": "안녕하세요! 주어진 차량 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

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
                            car_info = get_car_info(query)
                            if car_info is not None:
                                st.markdown("### 차량 정보")
                                st.dataframe(car_info)
                                car_number = car_info['사진번호'].values[0]
                                image_path = f"images/{car_number}.png"
                                logger.info(f"Looking for image at: {image_path}")
                                if os.path.exists(image_path):
                                    st.image(image_path, caption=f"차량 {car_number}")
                                else:
                                    st.markdown("이미지를 찾을 수 없습니다.")

                        with st.expander("참고 문서 확인"):
                            for doc in source_documents:
                                st.markdown(f"{doc.metadata['source']}", help=doc.page_content)

                        # Add assistant message to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            else:
                st.error("Conversation chain is not initialized. Please process the documents first.")

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):
    doc_list = []

    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()
        elif '.csv' in doc.name:
            df = pd.read_csv(file_name)
            if 'color_code' in df.columns:
                df = df.drop(columns=['color_code'])
            with NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
                df.to_csv(temp_file.name, index=False)
                temp_file.flush()
                loader = CSVLoader(file_path=temp_file.name)
                documents = loader.load()
                os.remove(temp_file.name)

        doc_list.extend(documents)
    return doc_list

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vetorestore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vetorestore.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )

    return conversation_chain

def save_uploaded_images(uploaded_images):
    if not os.path.exists("images"):
        os.makedirs("images")
    for img in uploaded_images:
        img_path = os.path.join("images", img.name)
        with open(img_path, "wb") as f:
            f.write(img.getvalue())
        logger.info(f"Uploaded image {img.name} to {img_path}")

def get_car_info(query):
    if st.session_state.car_data is not None:
        df = st.session_state.car_data
        # 예를 들어, 차량 이름으로 검색하는 경우
        result = df[df.apply(lambda row: query in row.to_string(), axis=1)]
        if not result.empty:
            return result
    return None

if __name__ == '__main__':
    main()
