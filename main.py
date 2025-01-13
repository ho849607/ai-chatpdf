import os
import shutil
import streamlit as st
from io import BytesIO
from dotenv import load_dotenv
import pdfplumber
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
import cv2
import numpy as np
import subprocess
import tempfile

# docx2txt 설치 확인
try:
    import docx2txt
    DOCX_ENABLED = True
except ImportError:
    DOCX_ENABLED = False

# PaddleOCR 설치 확인
try:
    from paddleocr import PaddleOCR
    PADDLE_OCR_ENABLED = True
    ocr = PaddleOCR(lang='ko')  
except ImportError:
    PADDLE_OCR_ENABLED = False

# NLTK
nltk.download('punkt')
nltk.download('stopwords')

korean_stopwords = [
    '이', '그', '저', '것', '수', '등', '들', '및', '더', '로', '를', '에',
    '의', '은', '는', '가', '와', '과', '하다', '있다', '되다', '이다',
    '으로', '에서', '까지', '부터', '까지', '만', '하다', '그리고',
    '하지만', '그러나'
]

dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    openai_api_key = st.sidebar.text_input("OpenAI API 키를 입력하세요.", type="password")
    if not openai_api_key:
        st.error("OpenAI API 키가 설정되지 않았습니다.")
        st.stop()

openai.api_key = openai_api_key

st.set_page_config(page_title="studyhelper")
st.title("studyhelper - (고급 버전 예시)")
st.write("---")

if 'lang' not in st.session_state:
    st.session_state.lang = 'english'

st.warning("저작물을 불법 복제하여 게시하는 경우 당사는 책임지지 않으며, 저작권법에 유의하세요.")

################################################################################
# (A) 새로 추가할 '문단별 중요도 평가 & chunking' 로직 예시
################################################################################
def chunk_text_by_heading(docx_text):
    """
    docx_text(전체 문자열)에서 'Heading1', 'Heading2' 등으로 분할하는 예시.
    실제로는 python-docx 등으로 문단 스타일을 확인해야 하지만,
    여기서는 간단히 '===Heading:' 마커를 가정해서 데모.
    """
    lines = docx_text.split('\n')
    chunks = []
    current_chunk = []
    heading_title = "NoHeading"
    chunk_id = 0

    for line in lines:
        if line.strip().startswith("===Heading:"):
            # 새 heading이 나오면, 이전 chunk를 저장
            if current_chunk:
                chunks.append({
                    "id": chunk_id,
                    "heading": heading_title,
                    "text": "\n".join(current_chunk)
                })
                chunk_id += 1
                current_chunk = []
            # heading 갱신
            heading_title = line.replace("===Heading:", "").strip()
        else:
            current_chunk.append(line)

    # 마지막 chunk 처리
    if current_chunk:
        chunks.append({
            "id": chunk_id,
            "heading": heading_title,
            "text": "\n".join(current_chunk)
        })
    return chunks

def gpt_evaluate_importance(chunk_text, language='korean'):
    """
    chunk_text를 GPT에게 주고 '중요도'를 1~5 사이로 분류, 요약 등 받는 예시.
    """
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    prompt = f"""
    아래 텍스트가 있습니다. 이 텍스트가 전체 문서에서 얼마나 중요한지 1~5 사이 정수로 판단해 주세요.
    그리고 요약도 1~2문장으로 해 주세요.

    텍스트:
    {chunk_text}

    응답 형식 예시:
    중요도: 4
    요약: ~~
    """
    messages = [HumanMessage(content=prompt)]
    response = llm(messages).content.strip()
    # 파싱 로직 단순 예시
    importance = 3
    short_summary = ""
    for line in response.split('\n'):
        if "중요도:" in line:
            try:
                importance = int(line.replace("중요도:", "").strip())
            except:
                pass
        if "요약:" in line:
            short_summary = line.replace("요약:", "").strip()

    return importance, short_summary

################################################################################
# (B) 기존 코드 일부 재사용
################################################################################

def add_chat_message(role, message):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({"role": role, "message": message})

def ask_gpt_question(question, language):
    llm = ChatOpenAI(
        model_name="gpt-4", 
        temperature=0, 
        streaming=True, 
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    if language == 'korean':
        prompt = f"다음 질문에 답변: {question}"
    else:
        prompt = question
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    return response.content

def chat_interface():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            with st.chat_message("user"):
                st.write(chat["message"])
        else:
            with st.chat_message("assistant"):
                st.write(chat["message"])

    if st.session_state.lang == 'korean':
        st.write("## ChatGPT와의 채팅 (GPT-4)")
        user_chat_input = st.chat_input("메시지를 입력하세요:")
    else:
        st.write("## Chat with ChatGPT (GPT-4)")
        user_chat_input = st.chat_input("Enter your message:")

    if user_chat_input:
        add_chat_message("user", user_chat_input)
        with st.chat_message("user"):
            st.write(user_chat_input)

        with st.spinner("GPT가 응답 중입니다..."):
            gpt_response = ask_gpt_question(user_chat_input, st.session_state.lang)
            add_chat_message("assistant", gpt_response)
            with st.chat_message("assistant"):
                st.write(gpt_response)

################################################################################
# 문서 포맷별 처리 함수 예시 (기존+부분 수정)
################################################################################

def docx_to_text(upload_file):
    if not DOCX_ENABLED:
        st.warning("docx2txt가 설치되어 있지 않아 .docx 파일을 처리할 수 없습니다.")
        return ""
    try:
        text = docx2txt.process(BytesIO(upload_file.getvalue()))
        return text if text else ""
    except Exception as e:
        st.error(f"DOCX 파일 처리 중 오류: {e}")
        return ""

################################################################################
# (C) chunking+중요도 평가 프로세스 적용 (DOCX 예시)
################################################################################
def docx_advanced_processing(docx_text):
    """
    1) 문단/heading 기준으로 chunk 분할
    2) GPT에 chunk별 중요도/요약 평가
    3) 성능 향상을 위한 placeholder 로직 (예: '단순 전처리 대비 20% 정확도 개선')
    4) 결괏값(각 chunk의 중요도, 요약)을 합쳐 최종 텍스트
    """
    chunks = chunk_text_by_heading(docx_text)
    combined_result = []
    for c in chunks:
        importance, short_summary = gpt_evaluate_importance(c["text"], language='korean')
        c["importance"] = importance
        c["short_summary"] = short_summary
        combined_result.append(c)
    
    # 성능 향상 placeholder
    # 실제론 별도 실험 결과가 필요
    st.write("**[실험 결과 가정]**: Chunking+중요도 분류를 적용했더니 일반 요약 대비 20% 정확도 향상!")

    # 예시로 최종 요약본 구성
    final_summary = []
    for c in combined_result:
        final_summary.append(f"[Heading: {c['heading']}] (중요도: {c['importance']})\n요약: {c['short_summary']}\n")

    # 텍스트 형태로 합침
    return "\n".join(final_summary)

################################################################################
# 나머지 함수(pdf_to_text, pptx_to_text, image_to_text 등)는 원본 코드 재활용
################################################################################

# ... pdf_to_text, pptx_to_text, image_to_text, etc.

################################################################################
# (D) 메인 로직
################################################################################
def main():
    uploaded_file = st.file_uploader(
        "파일을 업로드하세요 (PDF, PPTX, 이미지, HWP, DOC, DOCX)",
        type=['pdf', 'pptx', 'png', 'jpg', 'jpeg', 'hwp', 'doc', 'docx']
    )

    # GPT-4와의 채팅
    chat_interface()

    if uploaded_file is not None:
        filename = uploaded_file.name
        extension = os.path.splitext(filename)[1].lower()

        # 세션 중복 체크
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        if ("uploaded_file_hash" not in st.session_state or
            st.session_state.uploaded_file_hash != file_hash):
            st.session_state.uploaded_file_hash = file_hash
            st.session_state.extracted_text = ""
            st.session_state.summary = ""
            st.session_state.processed = False

        if not st.session_state.processed:
            if extension == ".docx":
                raw_text = docx_to_text(uploaded_file)
                # 고급 처리 (Chunking+GPT 중요도)
                if raw_text.strip():
                    advanced_summary = docx_advanced_processing(raw_text)
                    st.session_state.summary = advanced_summary
                    st.session_state.extracted_text = raw_text
                    st.success("DOCX 고급 분석 완료!")
                else:
                    st.error("DOCX에서 텍스트를 추출할 수 없습니다.")
                    st.session_state.summary = ""
            else:
                st.warning("현재 예시 코드에서는 .docx 고급처리만 구현되어 있습니다. 다른 포맷은 기존 로직 사용 등 확장 필요.")
            
            st.session_state.processed = True

        # 결과 표시
        if st.session_state.get("processed", False):
            if 'summary' in st.session_state and st.session_state.summary.strip():
                st.write("## (고급) 요약 결과")
                st.write(st.session_state.summary)
            else:
                st.write("## 요약 결과를 표시할 수 없습니다.")

if __name__ == "__main__":
    main()
