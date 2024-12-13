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

# NLTK 데이터 다운로드 설정
NLTK_DATA_DIR = os.path.expanduser("~/nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)
try:
    nltk.download('punkt', download_dir=NLTK_DATA_DIR)
    nltk.download('stopwords', download_dir=NLTK_DATA_DIR)
except FileExistsError:
    pass  # 이미 다운로드된 경우 무시

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
if 'gpt_questions' not in st.session_state:
    st.session_state.gpt_questions = []
if 'gpt_suggestions' not in st.session_state:
    st.session_state.gpt_suggestions = ""
if 'text_summary' not in st.session_state:
    st.session_state.text_summary = ""
if 'key_terms' not in st.session_state:
    st.session_state.key_terms = ""

st.title("📚 Study Helper with File Processing and Chat")
st.write("---")
st.warning("저작물을 불법 복제하여 게시하는 경우 당사는 책임지지 않으며, 저작권법에 유의하여 파일을 올려주세요.")

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
    try:
        response = llm(messages)
        language_code = response.content.strip().lower().split()[0]
        return language_code
    except Exception as e:
        st.error(f"언어 감지 중 오류가 발생했습니다: {e}")
        return "en"

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

# 요약 생성 함수
def generate_summary(extracted_text, language):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5, openai_api_key=openai_api_key)
    prompt = (
        f"다음 텍스트를 읽고 서론, 본론, 결론으로 구성된 요약을 작성해 주세요:\n\n{extracted_text}"
        if language == 'korean'
        else f"Read the following text and write a summary with introduction, body, and conclusion:\n\n{extracted_text}"
    )
    messages = [HumanMessage(content=prompt)]
    try:
        response = llm(messages)
        return response.content.strip()
    except openai.error.OpenAIError as e:
        st.error(f"요약 생성 중 오류가 발생했습니다: {e}")
        return ""

# 핵심 단어 분석 함수
def extract_key_terms(extracted_text, language):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5, openai_api_key=openai_api_key)
    prompt = (
        f"다음 텍스트에서 중요한 키워드 5~10개를 추출하고, 각 키워드의 간단한 설명을 작성해 주세요:\n\n{extracted_text}"
        if language == 'korean'
        else f"Extract 5 to 10 key terms from the text and provide a brief description for each:\n\n{extracted_text}"
    )
    messages = [HumanMessage(content=prompt)]
    try:
        response = llm(messages)
        return response.content.strip()
    except openai.error.OpenAIError as e:
        st.error(f"핵심 단어 분석 중 오류가 발생했습니다: {e}")
        return ""

# 텍스트 추출 및 처리 함수
def process_text(extracted_text):
    if not extracted_text.strip():
        st.error("파일에서 텍스트를 추출할 수 없습니다.")
        return

    st.session_state.extracted_text = extracted_text
    language_code = detect_language(extracted_text)
    st.session_state.lang = 'korean' if language_code == 'ko' else 'english'
    st.write(f"### 감지된 언어: {'한국어' if language_code == 'ko' else '영어'}")

    with st.spinner("GPT가 요약을 생성 중입니다..."):
        summary = generate_summary(extracted_text, st.session_state.lang)
        st.session_state.text_summary = summary

    with st.spinner("GPT가 핵심 단어를 분석 중입니다..."):
        key_terms = extract_key_terms(extracted_text, st.session_state.lang)
        st.session_state.key_terms = key_terms

    with st.spinner("GPT가 질문을 생성 중입니다..."):
        questions = generate_gpt_questions(extracted_text, st.session_state.lang)
        st.session_state.gpt_questions = questions

    with st.spinner("GPT가 제안 사항을 생성 중입니다..."):
        suggestions = generate_gpt_suggestions(extracted_text, st.session_state.lang)
        st.session_state.gpt_suggestions = suggestions

# GPT가 질문 생성 함수
def generate_gpt_questions(extracted_text, language):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5, openai_api_key=openai_api_key)
    prompt = (
        f"다음 텍스트를 기반으로 사용자가 깊이 생각할 수 있는 질문 3개를 제안해주세요:\n\n{extracted_text}"
        if language == 'korean'
        else f"Based on the following text, generate 3 thoughtful questions for deeper understanding:\n\n{extracted_text}"
    )
    messages = [HumanMessage(content=prompt)]
    try:
        response = llm(messages)
        questions = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
        return questions
    except openai.error.OpenAIError as e:
        st.error(f"질문 생성 중 오류가 발생했습니다: {e}")
        return []

# GPT가 제안 사항 생성 함수
def generate_gpt_suggestions(extracted_text, language):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5, openai_api_key=openai_api_key)
    prompt = (
        f"다음 텍스트에 기반하여 개선할 점과 주요 통찰을 제안해주세요:\n\n{extracted_text}"
        if language == 'korean'
        else f"Based on the following text, provide suggestions for improvement and key insights:\n\n{extracted_text}"
    )
    messages = [HumanMessage(content=prompt)]
    try:
        response = llm(messages)
        return response.content.strip()
    except openai.error.OpenAIError as e:
        st.error(f"제안 생성 중 오류가 발생했습니다: {e}")
        return ""

# 파일 업로드 처리
uploaded_file = st.file_uploader("파일을 올려주세요 (PDF, PPTX, PNG, JPG, JPEG, HWP 지원)", type=['pdf', 'pptx', 'png', 'jpg', 'jpeg', 'hwp'])
if uploaded_file:
    extracted_text = ""
    file_type = uploaded_file.type
    if file_type == 'application/pdf':
        try:
            with pdfplumber.open(BytesIO(uploaded_file.getvalue())) as pdf:
                extracted_text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        except Exception as e:
            st.error(f"PDF에서 텍스트 추출 중 오류: {e}")
    elif file_type == 'application/vnd.openxmlformats-officedocument.presentationml.presentation':
        try:
            prs = Presentation(BytesIO(uploaded_file.getvalue()))
            extracted_text = "\n".join([shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text")])
        except Exception as e:
            st.error(f"PPTX에서 텍스트 추출 중 오류: {e}")
    elif file_type.startswith('image/'):
        try:
            image = Image.open(uploaded_file)
            extracted_text = pytesseract.image_to_string(image, lang='kor+eng')
        except Exception as e:
            st.error(f"이미지에서 텍스트 추출 중 오류: {e}")
    elif file_type == 'application/haansoft-hwp':
        if HWP_SUPPORTED:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.hwp') as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                doc = pyhwp.HwpDocument(tmp_path)
                extracted_text = doc.body_text or ""
            except Exception as e:
                st.error(f"HWP 처리 중 오류: {e}")
        else:
            st.error("HWP 파일 처리를 지원하지 않습니다. pyhwp 라이브러리가 설치되어 있지 않습니다.")
    else:
        st.error("지원하지 않는 파일 형식입니다.")

    if extracted_text:
        process_text(extracted_text)

# 결과 출력
if st.session_state.text_summary:
    st.write("## 요약 결과")
    st.write(st.session_state.text_summary)

if st.session_state.key_terms:
    st.write("## 핵심 단어 분석")
    st.write(st.session_state.key_terms)

if st.session_state.gpt_questions:
    st.write("## GPT가 생성한 질문")
    for question in st.session_state.gpt_questions:
        st.write(f"- {question}")

if st.session_state.gpt_suggestions:
    st.write("## GPT의 제안 사항")
    st.write(st.session_state.gpt_suggestions)

# 사용자 입력 채팅 인터페이스
def chat_interface():
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.write(f"**사용자**: {chat['message']}")
        else:
            st.write(f"**GPT**: {chat['message']}")

    user_input = st.text_input("질문을 입력하세요:")
    if user_input:
        add_chat_message("user", user_input)
        with st.spinner("GPT가 응답 중입니다..."):
            response = ask_gpt_question(user_input, st.session_state.lang)
            add_chat_message("assistant", response)
            st.write(f"**GPT**: {response}")

    st.write("⚠️ ChatGPT는 실수할 수 있으며, 정보가 항상 정확하지 않을 수 있습니다. 중요한 내용을 확인하세요.")

chat_interface()
