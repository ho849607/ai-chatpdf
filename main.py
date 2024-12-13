import os
import streamlit as st
from io import BytesIO
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from pathlib import Path
import pdfplumber
from pptx import Presentation
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from PIL import Image
import pytesseract
import tempfile
import hashlib
from openai.error import RateLimitError, APIError
import time

# pyhwp 모듈 임포트 시도
try:
    import pyhwp
    HWP_SUPPORTED = True
except ModuleNotFoundError:
    HWP_SUPPORTED = False

# 초기 NLTK 데이터 다운로드
nltk.download('punkt')
nltk.download('stopwords')

# 한글 불용어 예시 (추가 가능)
korean_stopwords = [
    '이', '그', '저', '것', '수', '등', '들', '및', '더', '로', '를', '에',
    '의', '은', '는', '가', '와', '과', '하다', '있다', '되다', '이다',
    '으로', '에서', '까지', '부터', '까지', '만', '하다', '그리고',
    '하지만', '그러나'
]

# 환경 변수 로드
dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)
openai_api_key = os.getenv("OPENAI_API_KEY")

# Streamlit 초기 상태 설정
if 'lang' not in st.session_state:
    st.session_state.lang = 'english'
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""

# API 키 설정
if not openai_api_key:
    openai_api_key = st.sidebar.text_input("OpenAI API 키를 입력하세요.", type="password")
    if not openai_api_key:
        st.error("OpenAI API 키가 설정되지 않았습니다.")
        st.stop()
st.session_state["api_key"] = openai_api_key

# 페이지 타이틀 및 안내
st.title("📚 Study Helper with File Processing and Chat")
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

# 함수 정의
def add_chat_message(role, message):
    """채팅 기록에 메시지를 추가하는 함수"""
    st.session_state.chat_history.append({"role": role, "message": message})

def detect_language(text):
    """텍스트 언어 감지 함수"""
    if not text.strip():
        return "en"
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
    prompt = f"다음 텍스트의 언어를 ISO 639-1 코드로 감지해 주세요 (예: 'en'은 영어, 'ko'는 한국어):\n\n{text[:500]}"
    messages = [HumanMessage(content=prompt)]
    try:
        response = llm(messages)
        language_code = response.content.strip().lower().split()[0]
        return language_code
    except Exception as e:
        st.error(f"언어 감지 중 오류가 발생했습니다: {e}. 기본값(영어)을 사용합니다.")
        return "en"

def ask_gpt_question(question, language):
    """GPT에게 질문하고 답변을 반환하는 함수"""
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5, openai_api_key=openai_api_key)
    if language == 'korean':
        prompt = f"다음 질문에 답변: {question}"
    else:
        prompt = question
    messages = [HumanMessage(content=prompt)]
    try:
        response = llm(messages)
        return response.content.strip()
    except RateLimitError:
        st.error("API 호출이 제한되었습니다. 잠시 후 다시 시도하세요.")
        time.sleep(10)
        return ""
    except APIError as e:
        st.error(f"API 호출 중 오류가 발생했습니다: {e}")
        return "오류 발생: 작업을 완료하지 못했습니다."

def pdf_to_text(file_data):
    """PDF 파일에서 텍스트 추출"""
    try:
        with pdfplumber.open(BytesIO(file_data.getvalue())) as pdf:
            pages = []
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    pages.append(f"<PAGE{i+1}>\n{text}")
            return "\n".join(pages)
    except Exception as e:
        st.error(f"PDF에서 텍스트 추출 중 오류: {e}")
        return ""

def pptx_to_text(file_data):
    """PPTX 파일에서 텍스트 추출"""
    try:
        prs = Presentation(BytesIO(file_data.getvalue()))
        text_runs = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text_runs.append(shape.text)
        return "\n".join(text_runs)
    except Exception as e:
        st.error(f"PPTX에서 텍스트 추출 중 오류: {e}")
        return ""

def image_to_text(file_data):
    """이미지에서 텍스트 추출"""
    try:
        image = Image.open(file_data)
        text = pytesseract.image_to_string(image, lang='kor+eng')
        return text
    except Exception as e:
        st.error(f'이미지에서 텍스트 추출 중 오류: {e}')
        return ""

def hwp_or_hwpx_to_text(file_data, extension):
    """HWP/HWPX 파일에서 텍스트 추출"""
    if not HWP_SUPPORTED:
        st.error("HWP/HWPX 파일 처리를 지원하지 않습니다. pyhwp 라이브러리가 설치되어 있지 않습니다.")
        return ""
    if extension == '.hwpx':
        st.error("HWPX 파일은 현재 지원되지 않습니다.")
        return ""

    # HWP 파일 처리
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.hwp') as tmp:
            tmp.write(file_data.getvalue())
            tmp_path = tmp.name
        
        doc = pyhwp.HwpDocument(tmp_path)
        text = doc.body_text or ""
        return text
    except Exception as e:
        st.error(f"HWP 처리 중 오류: {e}")
        return ""

def process_text(extracted_text):
    """추출된 텍스트 언어 감지 및 상태 설정"""
    if not extracted_text.strip():
        st.error("파일에서 텍스트를 추출할 수 없습니다.")
        return
    else:
        st.success("텍스트 추출 완료.")
        language_code = detect_language(extracted_text)
        if language_code == 'ko':
            lang = 'korean'
            language_name = '한국어'
        elif language_code == 'en':
            lang = 'english'
            language_name = '영어'
        else:
            # 한국어 또는 영어가 아닌 경우 기본 영어로 진행
            lang = 'english'
            language_name = f'감지된 언어 코드: {language_code}, 기본 영어 사용'
        
        st.write(f"### 감지된 언어: {language_name}")
        st.session_state.lang = lang
        st.session_state.extracted_text = extracted_text

def chat_interface():
    """채팅 인터페이스 관리 함수"""
    # 기존 채팅 내역 표시
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            with st.chat_message("user"):
                st.write(chat["message"])
        else:
            with st.chat_message("assistant"):
                st.write(chat["message"])

    # 사용자 입력 받기
    if st.session_state.lang == 'korean':
        st.write("## ChatGPT와의 채팅")
        user_chat_input = st.chat_input("메시지를 입력하세요:")
    else:
        st.write("## Chat with ChatGPT")
        user_chat_input = st.chat_input("Enter your message:")

    # 사용자 메시지 처리
    if user_chat_input:
        add_chat_message("user", user_chat_input)
        with st.chat_message("user"):
            st.write(user_chat_input)

        # GPT 응답 처리
        with st.spinner("GPT가 응답 중입니다..."):
            gpt_response = ask_gpt_question(user_chat_input, st.session_state.lang)
            add_chat_message("assistant", gpt_response)
            with st.chat_message("assistant"):
                st.write(gpt_response)

# 파일 업로드 처리
uploaded_file = st.file_uploader("파일을 올려주세요 (PDF, PPTX, PNG, JPG, JPEG, HWP, HWPX 지원)",
                                 type=['pdf', 'pptx', 'png', 'jpg', 'jpeg', 'hwp', 'hwpx'])

if uploaded_file is not None:
    filename = uploaded_file.name
    extension = os.path.splitext(filename)[1].lower()
    extracted_text = ""

    if extension == ".pdf":
        extracted_text = pdf_to_text(uploaded_file)
    elif extension == ".pptx":
        extracted_text = pptx_to_text(uploaded_file)
    elif extension in [".png", ".jpg", ".jpeg"]:
        extracted_text = image_to_text(uploaded_file)
    elif extension in [".hwp", ".hwpx"]:
        extracted_text = hwp_or_hwpx_to_text(uploaded_file, extension)
    else:
        st.error("지원하지 않는 파일 형식입니다.")
    
    if extracted_text:
        process_text(extracted_text)

# 채팅 인터페이스 호출
chat_interface()
