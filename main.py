import os
import streamlit as st
from io import BytesIO
import hashlib
import openai
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.callbacks import StreamingStdOutCallbackHandler

# 문서 처리 라이브러리들
import pdfplumber  # PDF
from pptx import Presentation  # PPTX
import docx  # DOCX (python-docx)
import pytesseract  # 이미지 OCR
from PIL import Image
import tempfile
import subprocess

#######################################
# OpenAI API 키 설정
#######################################
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    openai.api_key = st.sidebar.text_input("OpenAI API 키를 입력하세요.", type="password")
    if not openai.api_key:
        st.error("OpenAI API 키가 설정되지 않았습니다.")
        st.stop()

#######################################
# 파일별 텍스트 추출 함수
#######################################

def pdf_to_text(file_bytes: bytes) -> str:
    """PDF → 텍스트"""
    text = ""
    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        st.error(f"PDF 텍스트 추출 중 오류: {e}")
    return text

def pptx_to_text(file_bytes: bytes) -> str:
    """PPTX → 텍스트"""
    text_runs = []
    try:
        prs = Presentation(BytesIO(file_bytes))
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text_runs.append(shape.text)
    except Exception as e:
        st.error(f"PPTX 텍스트 추출 중 오류: {e}")
    return "\n".join(text_runs)

def hwp_to_text(file_bytes: bytes) -> str:
    """HWP → 텍스트 (hwp5txt 필요)"""
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".hwp") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        result = subprocess.run(["hwp5txt", tmp_path], capture_output=True, text=True)
        if result.returncode == 0:
            text = result.stdout
        else:
            st.error("HWP에서 텍스트를 추출할 수 없습니다. (hwp5txt 설치 확인)")
    except FileNotFoundError:
        st.error("hwp5txt 실행 파일을 찾을 수 없습니다. 설치 후 PATH 환경변수를 확인하세요.")
    except Exception as e:
        st.error(f"HWP 처리 중 오류가 발생했습니다: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass
    return text

def docx_to_text(file_bytes: bytes) -> str:
    """DOCX → 텍스트 (python-docx)"""
    try:
        document = docx.Document(BytesIO(file_bytes))
        lines = [para.text for para in document.paragraphs]
        return "\n".join(lines)
    except Exception as e:
        st.error(f"DOCX 처리 중 오류: {e}")
        return ""

def doc_to_text(file_bytes: bytes) -> str:
    """
    DOC → DOCX 변환 후 텍스트 추출 (unoconv + LibreOffice 필요)
    """
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".doc") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    converted_path = tmp_path + ".docx"

    try:
        # unoconv를 이용해 DOC → DOCX 변환
        command = ["unoconv", "-f", "docx", tmp_path]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            st.error("unoconv로 .doc -> .docx 변환 실패")
            return ""
        # 변환된 파일 열기
        with open(converted_path, "rb") as f:
            converted_bytes = f.read()
        text = docx_to_text(converted_bytes)
    except FileNotFoundError:
        st.error("unoconv 또는 LibreOffice가 설치되어 있지 않아 .doc 파일을 처리할 수 없습니다.")
    except Exception as e:
        st.error(f"DOC 처리 중 오류: {e}")
    finally:
        # 임시 파일 정리
        try:
            os.remove(tmp_path)
        except:
            pass
        try:
            os.remove(converted_path)
        except:
            pass
    return text

def image_to_text(file_bytes: bytes) -> str:
    """이미지(OCR) → 텍스트 (pytesseract + Tesseract)"""
    text = ""
    # Tesseract 실행 파일 경로 맞게 설정
    pytesseract.pytesseract.tesseract_cmd = r"C:\Users\username\AppData\Local\Tesseract-OCR\tesseract.exe"

    try:
        with Image.open(BytesIO(file_bytes)) as img:
            text = pytesseract.image_to_string(img, lang="kor+eng")  # 한국어+영어
    except Exception as e:
        st.error(f"이미지 처리 중 오류: {e}")
    return text

#######################################
# GPT 요약/질문 함수
#######################################

def gpt_summarize(text: str, lang: str) -> str:
    """GPT로 핵심 내용 요약"""
    llm = ChatOpenAI(model_name="gpt-4", temperature=0, streaming=True,
                     callbacks=[StreamingStdOutCallbackHandler()])
    if lang == "korean":
        prompt = f"다음 텍스트를 간결하고 핵심 위주로 요약해 줘:\n\n{text}"
    else:
        prompt = f"Summarize the following text focusing on key points:\n\n{text}"
    response = llm([HumanMessage(content=prompt)])
    return response.content

def gpt_generate_questions(text: str, lang: str) -> str:
    """사용자가 놓칠 수 있는 관점을 질문으로 제시"""
    llm = ChatOpenAI(model_name="gpt-4", temperature=0, streaming=True,
                     callbacks=[StreamingStdOutCallbackHandler()])
    if lang == "korean":
        prompt = (
            f"다음 내용을 기반으로, 사용자가 놓칠 수 있는 핵심 관점이나 "
            f"추가 논의를 유발할 수 있는 질문 3~5개를 제시해줘:\n\n{text}"
        )
    else:
        prompt = (
            f"Based on the following content, generate 3-5 questions "
            f"that highlight key perspectives or prompt deeper discussion:\n\n{text}"
        )
    response = llm([HumanMessage(content=prompt)])
    return response.content

#######################################
# 간단한 언어 감지
#######################################
def detect_language(text: str) -> str:
    # 실제론 langdetect나 GPT 호출을 써도 됩니다.
    if any(ch in text for ch in ["은", "는", "이", "가", "을", "를", "하다", "한다"]):
        return "korean"
    return "english"

#######################################
# Streamlit 메인
#######################################

def main():
    st.title("자동 파일 분석 + ChatGPT (GPT-4)")
    st.write("---")

    # 업로드: PDF, PPTX, PNG/JPG/JPEG, HWP, DOC, DOCX
    uploaded_file = st.file_uploader(
        "파일을 업로드하세요 (PDF, PPTX, 이미지, HWP, DOC, DOCX)",
        type=["pdf","pptx","png","jpg","jpeg","hwp","doc","docx"]
    )

    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name
        st.write(f"**업로드된 파일:** {file_name}")

        # 확장자 구분
        _, ext = os.path.splitext(file_name)
        ext = ext.lower()

        # 추출된 텍스트
        extracted_text = ""

        # 파일 종류별로 분기
        if ext == ".pdf":
            extracted_text = pdf_to_text(file_bytes)
        elif ext == ".pptx":
            extracted_text = pptx_to_text(file_bytes)
        elif ext in [".png", ".jpg", ".jpeg"]:
            extracted_text = image_to_text(file_bytes)
        elif ext == ".hwp":
            extracted_text = hwp_to_text(file_bytes)
        elif ext == ".docx":
            extracted_text = docx_to_text(file_bytes)
        elif ext == ".doc":
            extracted_text = doc_to_text(file_bytes)
        else:
            st.error("지원하지 않는 파일 형식입니다.")
            return

        if not extracted_text.strip():
            st.error("파일에서 텍스트를 추출할 수 없습니다.")
            return

        # 언어 감지
        lang = detect_language(extracted_text)

        # GPT 요약
        st.subheader("파일 자동 요약")
        with st.spinner("GPT가 요약 중..."):
            summary = gpt_summarize(extracted_text, lang)
        st.write(summary)

        # GPT가 추가 질문 생성
        st.subheader("AI가 제안하는 추가 질문")
        with st.spinner("GPT가 질문 생성 중..."):
            questions = gpt_generate_questions(extracted_text, lang)
        st.write(questions)

        # 추가로 대화할 수 있는 인터페이스
        st.write("---")
        st.subheader("추가 대화")
        st.write("내용에 대해 더 궁금하신 점이 있으면 질문해 보세요.")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_input = st.text_input("질문 입력:")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            llm_chat = ChatOpenAI(model_name="gpt-4", temperature=0, streaming=True,
                                  callbacks=[StreamingStdOutCallbackHandler()])
            gpt_response = llm_chat([HumanMessage(content=user_input)]).content

            st.session_state.chat_history.append({"role": "assistant", "content": gpt_response})

        # 채팅 기록 표시
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"**[User]**: {msg['content']}")
            else:
                st.markdown(f"**[GPT]**: {msg['content']}")

if __name__ == "__main__":
    main()
