import os
import streamlit as st
from io import BytesIO
from dotenv import load_dotenv
import pdfplumber
# pptx(=python-pptx) 모듈이 필요합니다.
from pptx import Presentation

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.callbacks import StreamingStdOutCallbackHandler
import openai
from pathlib import Path
import hashlib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from PIL import Image

# pytesseract 모듈이 필요합니다.
import pytesseract

import subprocess
import tempfile

############################
# 초기 환경 설정
############################

# Tesseract 경로 (사용 환경에 맞춰 수정)
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\username\AppData\Local\Tesseract-OCR\tesseract.exe"

# NLTK 리소스 다운로드
nltk.download('punkt')
nltk.download('stopwords')

# 한국어 불용어 리스트 (필요 시 수정/추가)
korean_stopwords = [
    '이', '그', '저', '것', '수', '등', '들', '및', '더', '로', '를', '에',
    '의', '은', '는', '가', '와', '과', '하다', '있다', '되다', '이다',
    '으로', '에서', '까지', '부터', '만', '그리고', '하지만', '그러나'
]

# .env 파일에서 API 키 읽기
dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    openai_api_key = st.sidebar.text_input("OpenAI API 키를 입력하세요.", type="password")
    if not openai_api_key:
        st.error("OpenAI API 키가 설정되지 않았습니다.")
        st.stop()

openai.api_key = openai_api_key

############################
# Streamlit UI
############################

st.title("📚 Study Helper (GPT-4)")
st.write("---")

if 'lang' not in st.session_state:
    st.session_state.lang = 'english'

st.warning("저작물을 불법 복제·게시하면 책임지지 않으며, 저작권법에 유의해주세요.")

############################
# 주요 함수
############################

def add_chat_message(role, message):
    """채팅 메시지를 세션에 기록"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({"role": role, "message": message})

def ask_gpt_question(question, language):
    """ChatOpenAI(GPT-4)로 질문"""
    llm = ChatOpenAI(
        model_name="gpt-4", 
        temperature=0, 
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    if language == 'korean':
        prompt = f"다음 질문에 답변해 주세요:\n\n{question}"
    else:
        prompt = question

    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    return response.content

def chat_interface():
    """ChatGPT처럼 대화할 수 있는 인터페이스"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 기존 채팅 기록 표시
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            with st.chat_message("user"):
                st.write(chat["message"])
        else:
            with st.chat_message("assistant"):
                st.write(chat["message"])

    # 대화 입력창
    if st.session_state.lang == 'korean':
        st.write("## ChatGPT와의 채팅 (GPT-4)")
        user_chat_input = st.chat_input("메시지를 입력하세요:")
    else:
        st.write("## Chat with ChatGPT (GPT-4)")
        user_chat_input = st.chat_input("Enter your message:")

    # 사용자가 입력하면 GPT 응답
    if user_chat_input:
        add_chat_message("user", user_chat_input)
        with st.chat_message("user"):
            st.write(user_chat_input)

        with st.spinner("GPT가 응답 중입니다..."):
            gpt_response = ask_gpt_question(user_chat_input, st.session_state.lang)
            add_chat_message("assistant", gpt_response)
            with st.chat_message("assistant"):
                st.write(gpt_response)

def pdf_to_text(upload_file):
    """PDF에서 텍스트 추출"""
    try:
        with pdfplumber.open(BytesIO(upload_file.getvalue())) as pdf:
            pages = []
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    pages.append(f"<PAGE{i+1}>\n{text}")
            return "\n".join(pages)
    except Exception as e:
        st.error(f"PDF에서 텍스트를 추출하는 중 오류가 발생했습니다: {e}")
        return ""

def pptx_to_text(upload_file):
    """PPTX에서 텍스트 추출"""
    try:
        prs = Presentation(BytesIO(upload_file.getvalue()))
        text_runs = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text_runs.append(shape.text)
        return "\n".join(text_runs)
    except Exception as e:
        st.error(f"PPTX에서 텍스트를 추출하는 중 오류가 발생했습니다: {e}")
        return ""

def image_to_text(uploaded_image):
    """이미지(캡처본 등)에서 텍스트(OCR) 추출"""
    try:
        image = Image.open(uploaded_image)
        # Tesseract 설치 및 경로 확인
        if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
            st.error("Tesseract가 설치되어 있지 않거나 경로가 올바르지 않습니다.")
            return ""
        # 'kor+eng'로 한국어+영어 혼합 인식 가능
        text = pytesseract.image_to_string(image, lang='kor+eng')
        return text
    except Exception as e:
        st.error(f'이미지에서 텍스트를 추출 중 오류가 발생했습니다: {e}')
        return ""

############################
# 메인 로직 (예시)
############################

def main():
    # 파일 업로더
    uploaded_file = st.file_uploader("파일을 업로드해 주세요 (PDF, PPTX, PNG, JPG, JPEG 등)")

    # ChatGPT 대화 인터페이스
    chat_interface()

    if uploaded_file:
        filename = uploaded_file.name
        st.write(f"업로드된 파일 이름: {filename}")

        # 여기서 pptx_to_text, image_to_text 등을 호출하여 텍스트를 추출하는 로직 추가 가능
        # ...

if __name__ == "__main__":
    main()
