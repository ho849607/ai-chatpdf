import os
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
import pytesseract
import subprocess
import tempfile
import docx2txt  # docx 처리 라이브러리 (pip install docx2txt)

# Tesseract 경로 (실행 환경에 맞게 수정 필요)
pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"

# NLTK 리소스 다운로드
nltk.download('punkt')
nltk.download('stopwords')

# 한국어 불용어 리스트 (필요하면 수정/추가)
korean_stopwords = [
    '이', '그', '저', '것', '수', '등', '들', '및', '더', '로', '를', '에',
    '의', '은', '는', '가', '와', '과', '하다', '있다', '되다', '이다',
    '으로', '에서', '까지', '부터', '까지', '만', '하다', '그리고',
    '하지만', '그러나'
]

# .env 파일에서 환경 변수 로드
dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)

# OpenAI API 키 설정
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    openai_api_key = st.sidebar.text_input("OpenAI API 키를 입력하세요.", type="password")
    if not openai_api_key:
        st.error("OpenAI API 키가 설정되지 않았습니다.")
        st.stop()

openai.api_key = openai_api_key

# ------------------------ 제목을 studyhelper로 변경 ------------------------
st.title("studyhelper")
st.write("---")

if 'lang' not in st.session_state:
    st.session_state.lang = 'english'

st.warning("저작물을 불법 복제하여 게시하는 경우 당사는 책임지지 않으며, 저작권법에 유의하여 파일을 올려주세요.")

def add_chat_message(role, message):
    """채팅 히스토리를 세션에 저장하는 헬퍼 함수"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({"role": role, "message": message})

def ask_gpt_question(question, language):
    """GPT에게 질문하고 답변을 반환"""
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
    """화면에 채팅 인터페이스 구성"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 기존 채팅 히스토리 표시
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            with st.chat_message("user"):
                st.write(chat["message"])
        else:
            with st.chat_message("assistant"):
                st.write(chat["message"])

    # 사용자 입력
    if st.session_state.lang == 'korean':
        st.write("## ChatGPT와의 채팅 (GPT-4)")
        user_chat_input = st.chat_input("메시지를 입력하세요:")
    else:
        st.write("## Chat with ChatGPT (GPT-4)")
        user_chat_input = st.chat_input("Enter your message:")

    # 사용자 입력을 처리
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
    """PDF 파일에서 텍스트 추출"""
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
    """PPTX 파일에서 텍스트 추출"""
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
    """이미지 파일에서 텍스트 추출 (pytesseract 사용)"""
    try:
        image = Image.open(uploaded_image)
        if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
            st.error("Tesseract가 설치되어 있지 않거나 경로가 올바르지 않습니다.")
            return ""
        text = pytesseract.image_to_string(image, lang='kor+eng')
        return text
    except Exception as e:
        st.error(f'이미지에서 텍스트를 추출하는 중 오류가 발생했습니다: {e}')
        return ""

def hwp_to_text(upload_file):
    """HWP 파일에서 텍스트 추출 (hwp5txt 사용)"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.hwp') as tmp:
            tmp.write(upload_file.getvalue())
            tmp_path = tmp.name
        result = subprocess.run(["hwp5txt", tmp_path], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            st.error("HWP에서 텍스트를 추출할 수 없습니다. hwp5txt 툴이 설치되어 있는지 확인해주세요.")
            return ""
    except FileNotFoundError:
        st.error("hwp5txt 명령어를 찾을 수 없습니다. hwp5txt가 제대로 설치되어 PATH에 포함되어 있는지 확인해주세요.")
        return ""
    except Exception as e:
        st.error(f"HWP 처리 중 오류가 발생했습니다: {e}")
        return ""

def docx_to_text(upload_file):
    """DOCX 파일에서 텍스트 추출 (docx2txt 사용)"""
    try:
        text = docx2txt.process(BytesIO(upload_file.getvalue()))
        return text if text else ""
    except Exception as e:
        st.error(f"DOCX 파일 처리 중 오류가 발생했습니다: {e}")
        return ""

def doc_to_text(upload_file):
    """DOC 파일에서 텍스트 추출 (antiword 필요)"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.doc') as tmp:
            tmp.write(upload_file.getvalue())
            tmp_path = tmp.name
        result = subprocess.run(["antiword", tmp_path], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            st.error("DOC에서 텍스트를 추출할 수 없습니다. antiword 툴이 설치되어 있는지 확인해주세요.")
            return ""
    except FileNotFoundError:
        st.error("antiword 명령어를 찾을 수 없습니다. antiword가 제대로 설치되어 PATH에 포함되어 있는지 확인해주세요.")
        return ""
    except Exception as e:
        st.error(f"DOC 처리 중 오류가 발생했습니다: {e}")
        return ""

def detect_language(text):
    """업로드된 텍스트의 언어를 ISO 639-1 코드로 감지"""
    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    prompt = f"다음 텍스트의 언어를 ISO 639-1 코드로 감지해 주세요 (예: 'en'은 영어, 'ko'는 한국어):\n\n{text[:500]}"
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    language_code = response.content.strip().lower().split()[0]
    return language_code

def summarize_text(text, language):
    """추출 텍스트 요약 (서론, 본론, 결론)"""
    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    if language == 'korean':
        prompt = f"다음 텍스트를 읽고 서론, 본론, 결론으로 구성된 자세한 요약을 작성해 주세요:\n\n{text}"
    else:
        prompt = f"Read the following text and write a detailed summary with introduction, main body, and conclusion:\n\n{text}"
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    return response.content.strip()

def extract_key_summary_words_with_sources(text, language):
    """키워드 추출 (5~10개) + 출처 표시"""
    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    if language == 'korean':
        prompt = f"""다음 텍스트에서 중요한 키워드 5~10개를 추출하고, 각 키워드의 출처를 표시해주세요.

키워드1 (출처)
키워드2 (출처)
...

텍스트:
{text}"""
    else:
        prompt = f"""Extract 5 to 10 important keywords from the text and indicate their sources:

Keyword1 (Source)
Keyword2 (Source)
...

Text:
{text}"""
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    return response.content.strip()

def extract_and_search_terms(summary_text, extracted_text, language='english'):
    """요약에서 중요한 용어 5~10개를 추출 후 정의/페이지 정보 제공"""
    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    if language == 'korean':
        prompt = f"다음 요약에서 중요한 용어 5~10개를 추출하고, 각 용어 정의와 텍스트 내 페이지 정보를 제공:\n\n{summary_text}"
    else:
        prompt = f"From the following summary, extract 5-10 important terms, provide detailed definitions and their page references:\n\n{summary_text}"
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    return response.content.strip()

# 자동 검색 + 설명: 찾은 문맥을 LLM에 전달해 추가 정보를 요약해 주는 함수
def search_and_auto_explain(text, search_query, language='english'):
    """사용자가 입력한 키워드로 텍스트를 검색하고,
       해당 문맥을 GPT에 전달하여 요약/설명을 생성해주는 기능"""
    # 키워드 포함 문장(혹은 라인) 검색
    results = []
    for line in text.split('\n'):
        if search_query.lower() in line.lower():
            results.append(line.strip())

    # 결과가 없다면 바로 반환
    if not results:
        return None, None

    # 검색된 문맥을 일정 길이로 묶어서 GPT에 전달 (라인이 많을 경우 대비)
    # 여기서는 간단히 전부 합쳐서 전달하지만, 필요하다면 토큰 조절/분할 로직 사용 가능
    matched_text = "\n".join(results)

    # GPT에게 추가 설명/요약 요청
    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )

    if language == 'korean':
        prompt = f"""다음 텍스트는 사용자가 '{search_query}'를 포함하는 문맥입니다.
이 문맥에 대해 간단한 요약 또는 추가 설명을 해주세요.

문맥:
{matched_text}
"""
    else:
        prompt = f"""The following text contains the search term '{search_query}'.
Please provide a brief summary or explanation about this content.

Matched context:
{matched_text}
"""

    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    explanation = response.content.strip()

    return results, explanation

def generate_questions_for_user(text, language):
    """사용자가 더 깊이 생각할 수 있는 3개 질문 생성"""
    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    if language == 'korean':
        prompt = f"다음 내용을 기반으로 사용자가 깊이 생각할 수 있는 질문 3개 제시:\n\n{text}"
    else:
        prompt = f"Based on the following content, generate 3 thoughtful questions for deeper understanding:\n\n{text}"
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    questions = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
    return questions

def create_ppt_from_text(text, filename="summary_output.pptx"):
    """요약 내용을 PPT 파일로 변환 후 다운로드할 수 있는 객체 생성"""
    prs = Presentation()
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    slide.shapes.title.text = "Summary"
    slide.placeholders[1].text = text

    buf = BytesIO()
    prs.save(buf)
    buf.seek(0)
    return buf

# 세션 초기화
if "processed" not in st.session_state:
    st.session_state.processed = False

# 파일 업로더 (PDF, PPTX, PNG, JPG, JPEG, HWP, DOC, DOCX 지원)
uploaded_file = st.file_uploader(
    "파일을 업로드하세요 (PDF, PPTX, 이미지, HWP, DOC, DOCX)",
    type=['pdf', 'pptx', 'png', 'jpg', 'jpeg', 'hwp', 'doc', 'docx']
)

# GPT-4와의 채팅 인터페이스
chat_interface()

if uploaded_file is not None:
    filename = uploaded_file.name
    extension = os.path.splitext(filename)[1].lower()

    # 업로드된 파일 해시로 중복 처리 방지
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    if ("uploaded_file_hash" not in st.session_state or
        st.session_state.uploaded_file_hash != file_hash):
        # 새 파일이 업로드된 경우 세션 초기화
        st.session_state.uploaded_file_hash = file_hash
        st.session_state.extracted_text = ""
        st.session_state.summary = ""
        st.session_state.keywords = ""
        st.session_state.term_info = ""
        st.session_state.gpt_questions = []
        st.session_state.processed = False

    if not st.session_state.processed:
        # 확장자별 텍스트 추출
        if extension == ".pdf":
            extracted_text = pdf_to_text(uploaded_file)
        elif extension == ".pptx":
            extracted_text = pptx_to_text(uploaded_file)
        elif extension in [".png", ".jpg", ".jpeg"]:
            extracted_text = image_to_text(uploaded_file)
        elif extension == ".hwp":
            extracted_text = hwp_to_text(uploaded_file)
        elif extension == ".docx":
            extracted_text = docx_to_text(uploaded_file)
        elif extension == ".doc":
            extracted_text = doc_to_text(uploaded_file)
        else:
            st.error("지원하지 않는 파일 형식입니다. PDF, PPTX, PNG, JPG, JPEG, HWP, DOC, DOCX만 업로드하세요.")
            extracted_text = ""

        # 텍스트가 추출되지 않았을 경우
        if not extracted_text.strip():
            st.error("업로드된 파일에서 텍스트를 추출할 수 없습니다.")
            st.session_state.summary = ""
        else:
            st.success("텍스트 추출 완료!")

            # 언어 감지
            language_code = detect_language(extracted_text)
            if language_code == 'ko':
                lang = 'korean'
                language_name = '한국어'
            elif language_code == 'en':
                lang = 'english'
                language_name = '영어'
            else:
                lang = 'english'
                language_name = '알 수 없음 (영어 진행)'

            st.write(f"### 감지된 언어: {language_name}")
            st.session_state.lang = lang
            st.session_state.extracted_text = extracted_text

            # 요약
            with st.spinner("요약 생성 중..."):
                summary = summarize_text(extracted_text, lang)
                st.session_state.summary = summary

            # 핵심 단어
            with st.spinner("핵심 단어 추출 중..."):
                key_summary_words = extract_key_summary_words_with_sources(extracted_text, lang)
                st.session_state.keywords = key_summary_words

            # 중요 단어 정보
            with st.spinner("중요 단어 정보 추출 중..."):
                term_info = extract_and_search_terms(summary, extracted_text, language=lang)
                st.session_state.term_info = term_info

            # GPT가 사용자에게 질문
            with st.spinner("GPT가 질문을 생성 중..."):
                gpt_questions = generate_questions_for_user(extracted_text, lang)
                st.session_state.gpt_questions = gpt_questions

            st.session_state.processed = True

    # 처리 후 결과 표시
    if st.session_state.get("processed", False):
        if 'summary' in st.session_state and st.session_state.summary.strip():
            st.write("## 요약 결과")
            st.write(st.session_state.summary)
        else:
            st.write("## 요약 결과를 표시할 수 없습니다.")

        if 'keywords' in st.session_state and st.session_state.keywords.strip():
            st.write("## 핵심 요약 단어 및 출처")
            st.write(st.session_state.keywords)

        if 'term_info' in st.session_state and st.session_state.term_info.strip():
            st.write("## 요약 내 중요한 단어 정보")
            st.write(st.session_state.term_info)

        st.write("---")
        # PPT 다운로드
        if st.button("요약 내용을 PPT로 다운로드"):
            ppt_buffer = create_ppt_from_text(st.session_state.summary)
            st.download_button(
                label="PPT 다운로드",
                data=ppt_buffer,
                file_name="summary_output.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
            )

# ----------------------------- 자동 검색 + 설명 기능 -----------------------------
if st.session_state.get("processed", False):
    st.write("---")
    if st.session_state.lang == 'korean':
        st.write("## 🔍 키워드 검색 및 자동 정보 제공")
        search_query = st.text_input("검색할 키워드를 입력하세요:")
    else:
        st.write("## 🔍 Keyword Search & Auto Explanation")
        search_query = st.text_input("Enter a keyword to search:")

    if search_query:
        with st.spinner("검색 중..."):
            results, explanation = search_and_auto_explain(st.session_state.extracted_text, search_query, st.session_state.lang)
        if results:
            # 검색 결과 표시
            if st.session_state.lang == 'korean':
                st.write("### 검색된 문맥:")
            else:
                st.write("### Matched Context:")
            for r in results:
                st.write(f"- {r}")
            # GPT가 생성한 추가 설명 표시
            if explanation:
                if st.session_state.lang == 'korean':
                    st.write("### GPT가 제공하는 추가 정보/설명:")
                else:
                    st.write("### GPT's Additional Info/Explanation:")
                st.write(explanation)
        else:
            if st.session_state.lang == 'korean':
                st.write("검색 결과가 없습니다.")
            else:
                st.write("No results found.")

# GPT가 사용자에게 질문 -> 사용자 답변에 대해 GPT 피드백
if st.session_state.get("processed", False):
    st.write("---")
    if st.session_state.lang == 'korean':
        st.write("## GPT가 당신에게 질문합니다")
    else:
        st.write("## GPT has questions for you")

    if "gpt_questions" in st.session_state:
        for idx, question in enumerate(st.session_state.gpt_questions):
            user_answer = st.text_input(f"**{question}**", key=f"gpt_question_{idx}")
            if user_answer:
                with st.spinner("GPT가 응답을 검토 중입니다..."):
                    if st.session_state.lang == 'korean':
                        feedback_prompt = (
                            f"{question}\n\n사용자 답변: {user_answer}\n\n"
                            "이 답변에 대한 피드백을 제공해 주세요."
                        )
                    else:
                        feedback_prompt = (
                            f"{question}\n\nUser's answer: {user_answer}\n\n"
                            "Please provide feedback on this answer."
                        )
                    feedback = ask_gpt_question(feedback_prompt, st.session_state.lang)
                    if st.session_state.lang == 'korean':
                        st.write("### GPT의 피드백")
                    else:
                        st.write("### GPT's Feedback")
                    st.write(feedback)

st.write("---")
st.info("**⚠ ChatGPT는 때때로 부정확하거나 오해의 소지가 있는 답변을 할 수 있습니다. 중요한 정보는 추가로 검증하세요.**")
