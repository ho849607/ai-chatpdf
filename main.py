import os
import streamlit as st
from io import BytesIO
from dotenv import load_dotenv
import pdfplumber
from pptx import Presentation
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import openai
from pathlib import Path
import hashlib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from PIL import Image
import pytesseract
import tempfile
import time

# pyhwp 모듈 임포트 시도
try:
    import pyhwp
    HWP_SUPPORTED = True
except ModuleNotFoundError:
    HWP_SUPPORTED = False

# 초기 설정
nltk.download('punkt')
nltk.download('stopwords')

# 환경 변수 로드
dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    openai_api_key = st.sidebar.text_input("OpenAI API 키를 입력하세요.", type="password")
    if not openai_api_key:
        st.error("OpenAI API 키가 설정되지 않았습니다.")
        st.stop()
openai.api_key = openai_api_key

# Streamlit 초기 상태 설정
if 'lang' not in st.session_state:
    st.session_state.lang = 'english'
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""

st.title("\ud83d\udcda Study Helper with File Processing and Chat")
st.write("---")

st.warning("저작물을 불법 복제하여 게시하는 경우 당사는 책임지지 않으며, 저작권법에 유의하여 파일을 올려주세요.")

# 사이드바: 기록 보관 기능
st.sidebar.write("## 기록 보관")
if st.session_state.chat_history:
    chat_text = "\n".join([f"{msg['role']}: {msg['message']}" for msg in st.session_state.chat_history])
    st.sidebar.download_button(
        "채팅 기록 다운로드",
        data=chat_text.encode('utf-8'),
        file_name="chat_history.txt",
        mime="text/plain"
    )
else:
    st.sidebar.write("채팅 기록이 없습니다.")

# 채팅 기록 추가 함수
def add_chat_message(role, message):
    st.session_state.chat_history.append({"role": role, "message": message})

# 언어 감지 함수
def detect_language(text):
    if not text.strip():
        return "en"
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
    prompt = f"다음 텍스트의 언어를 ISO 639-1 코드로 감지해 주세요 (예: 'en'은 영어, 'ko'는 한국어):\n\n{text[:500]}"
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    language_code = response.content.strip().lower().split()[0]
    return language_code

# ChatGPT 응답 함수
def ask_gpt_question(question, language):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5, openai_api_key=openai_api_key)
    prompt = question if language == 'english' else f"다음 질문에 답변: {question}"
    messages = [HumanMessage(content=prompt)]
    try:
        response = llm(messages)
        return response.content.strip()
    except openai.error.OpenAIError as e:
        st.error(f"API 호출 중 오류가 발생했습니다: {e}")
        return ""

# 텍스트 추출 함수
def pdf_to_text(file_data):
    try:
        with pdfplumber.open(BytesIO(file_data.getvalue())) as pdf:
            return "\n".join([page.extract_text() or "" for page in pdf.pages])
    except Exception as e:
        st.error(f"PDF에서 텍스트 추출 중 오류: {e}")
        return ""

def pptx_to_text(file_data):
    try:
        prs = Presentation(BytesIO(file_data.getvalue()))
        return "\n".join([shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text")])
    except Exception as e:
        st.error(f"PPTX에서 텍스트 추출 중 오류: {e}")
        return ""

def image_to_text(file_data):
    try:
        image = Image.open(file_data)
        return pytesseract.image_to_string(image, lang='kor+eng')
    except Exception as e:
        st.error(f"이미지에서 텍스트 추출 중 오류: {e}")
        return ""

def hwp_to_text(file_data):
    if not HWP_SUPPORTED:
        st.error("HWP 파일 처리를 지원하지 않습니다. pyhwp 라이브러리가 설치되어 있지 않습니다.")
        return ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.hwp') as tmp:
            tmp.write(file_data.getvalue())
            tmp_path = tmp.name
        doc = pyhwp.HwpDocument(tmp_path)
        return doc.body_text or ""
    except Exception as e:
        st.error(f"HWP 처리 중 오류: {e}")
        return ""

# 텍스트 처리 함수
def process_text(extracted_text):
    if not extracted_text.strip():
        st.error("파일에서 텍스트를 추출할 수 없습니다.")
        return

    st.session_state.extracted_text = extracted_text
    language_code = detect_language(extracted_text)
    st.session_state.lang = 'korean' if language_code == 'ko' else 'english'
    st.write(f"### 감지된 언어: {'한국어' if language_code == 'ko' else '영어'}")

# 파일 업로드 처리
uploaded_file = st.file_uploader("파일을 올려주세요 (PDF, PPTX, PNG, JPG, JPEG, HWP 지원)", type=['pdf', 'pptx', 'png', 'jpg', 'jpeg', 'hwp'])
if uploaded_file:
    file_type = uploaded_file.type
    if file_type == 'application/pdf':
        extracted_text = pdf_to_text(uploaded_file)
    elif file_type == 'application/vnd.openxmlformats-officedocument.presentationml.presentation':
        extracted_text = pptx_to_text(uploaded_file)
    elif file_type.startswith('image/'):
        extracted_text = image_to_text(uploaded_file)
    elif file_type == 'application/haansoft-hwp':
        extracted_text = hwp_to_text(uploaded_file)
    else:
        st.error("지원하지 않는 파일 형식입니다.")
        extracted_text = ""

    if extracted_text:
        process_text(extracted_text)

# 채팅 인터페이스
chat_interface()

