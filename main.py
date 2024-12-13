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
import openai  # openai 임포트
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

# 환경 변수 로드
dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)
openai_api_key = os.getenv("OPENAI_API_KEY")

if 'lang' not in st.session_state:
    st.session_state.lang = 'english'
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'last_user_input' not in st.session_state:
    st.session_state.last_user_input = ""

if not openai_api_key:
    openai_api_key = st.sidebar.text_input("OpenAI API 키를 입력하세요.", type="password")
    if not openai_api_key:
        st.error("OpenAI API 키가 설정되지 않았습니다.")
        st.stop()
st.session_state["api_key"] = openai_api_key

st.title("📚 Study Helper with File Processing and Chat")
st.write("---")
st.warning("저작물을 불법 복제하여 게시하는 경우 당사는 책임지지 않으며, 저작권법에 유의하여 파일을 올려주세요.")

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

def add_chat_message(role, message):
    st.session_state.chat_history.append({"role": role, "message": message})

def detect_language(text):
    if not text.strip():
        return "en"
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=st.session_state["api_key"])
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
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5, openai_api_key=st.session_state["api_key"])
    if language == 'korean':
        prompt = f"다음 질문에 답변: {question}"
    else:
        prompt = question
    messages = [HumanMessage(content=prompt)]
    try:
        response = llm(messages)
        return response.content.strip()
    except openai.error.RateLimitError:
        st.error("API 호출이 제한되었습니다. 잠시 후 다시 시도하세요.")
        time.sleep(10)
        return ""
    except openai.error.APIError as e:
        st.error(f"API 호출 중 오류가 발생했습니다: {e}")
        return "오류 발생: 작업을 완료하지 못했습니다."

def suggest_improvements(user_input, reference_text, language):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5, openai_api_key=st.session_state["api_key"])
    if language == 'korean':
        prompt = (
            f"다음은 참고 텍스트입니다:\n{reference_text}\n\n"
            f"사용자의 응답: {user_input}\n\n"
            f"참고 텍스트를 기준으로 사용자의 응답에 대한 피드백을 제공해주세요. "
            f"응답의 맞는 부분과 잘못된 부분을 지적하고, 어떻게 하면 더 정확하거나 개선된 응답을 할 수 있을지 제안해 주세요."
        )
    else:
        prompt = (
            f"Here is the reference text:\n{reference_text}\n\n"
            f"User's answer: {user_input}\n\n"
            f"Based on the reference text, please provide feedback on the user's answer. "
            f"Point out what is correct and incorrect, and suggest how to improve or refine the answer."
        )

    messages = [HumanMessage(content=prompt)]
    try:
        response = llm(messages)
        return response.content.strip()
    except:
        return "개선 제안을 제공할 수 없습니다."

def pdf_to_text(file_data):
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
    try:
        image = Image.open(file_data)
        text = pytesseract.image_to_string(image, lang='kor+eng')
        return text
    except Exception as e:
        st.error(f'이미지에서 텍스트 추출 중 오류: {e}')
        return ""

def hwp_or_hwpx_to_text(file_data, extension):
    if not HWP_SUPPORTED:
        st.error("HWP/HWPX 파일 처리를 지원하지 않습니다. pyhwp 라이브러리가 설치되어 있지 않습니다.")
        return ""
    if extension == '.hwpx':
        st.error("HWPX 파일은 현재 지원되지 않습니다.")
        return ""

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
            lang = 'english'
            language_name = f'감지된 언어 코드: {language_code}, 기본 영어 사용'
        
        st.write(f"### 감지된 언어: {language_name}")
        st.session_state.lang = lang
        st.session_state.extracted_text = extracted_text

def chat_interface():
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            with st.chat_message("user"):
                st.write(chat["message"])
        else:
            with st.chat_message("assistant"):
                st.write(chat["message"])

    if st.session_state.lang == 'korean':
        st.write("## ChatGPT와의 채팅")
        user_chat_input = st.chat_input("메시지를 입력하세요:")
    else:
        st.write("## Chat with ChatGPT")
        user_chat_input = st.chat_input("Enter your message:")

    if user_chat_input:
        st.session_state.last_user_input = user_chat_input
        add_chat_message("user", user_chat_input)
        with st.chat_message("user"):
            st.write(user_chat_input)

        with st.spinner("GPT가 응답 중입니다..."):
            gpt_response = ask_gpt_question(user_chat_input, st.session_state.lang)
            add_chat_message("assistant", gpt_response)
            with st.chat_message("assistant"):
                st.write(gpt_response)

        if st.session_state.extracted_text.strip():
            improvement_suggestions = suggest_improvements(
                st.session_state.last_user_input,
                st.session_state.extracted_text,
                st.session_state.lang
            )
            with st.chat_message("assistant"):
                st.write("### 개선 사항 및 추천")
                st.write(improvement_suggestions)

uploaded_file = st.file_uploader(
    "파일을 올려주세요 (PDF, PPTX, PNG, JPG, JPEG, HWP, HWPX 지원)",
    type=['pdf', 'pptx', 'png', 'jpg', 'jpeg', 'hwp', 'hwpx']
)

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

chat_interface()
