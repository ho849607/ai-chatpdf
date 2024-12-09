import os
import streamlit as st
from io import BytesIO
from dotenv import load_dotenv
import pdfplumber
from pptx import Presentation  # PPTX 파일 처리를 위한 라이브러리
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
import subprocess  # hwp 처리용
import tempfile

# Tesseract 경로 설정 (Windows 환경에서 Tesseract-OCR을 설치한 경로로 변경)
# 예: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# 아래는 필요시 주석 해제 후 사용하세요.
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 초기 설정
nltk.download('punkt')
nltk.download('stopwords')

# 한국어 불용어 리스트 정의
korean_stopwords = ['이', '그', '저', '것', '수', '등', '들', '및', '더', '로', '를', '에',
                    '의', '은', '는', '가', '와', '과', '하다', '있다', '되다', '이다',
                    '으로', '에서', '까지', '부터', '까지', '만', '하다', '그리고',
                    '하지만', '그러나']

# .env 파일에서 환경 변수 로드
dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)

# API 키 설정
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    openai_api_key = st.sidebar.text_input("OpenAI API 키를 입력하세요.", type="password")
    if not openai_api_key:
        st.error("OpenAI API 키가 설정되지 않았습니다.")
        st.stop()

openai.api_key = openai_api_key

# 제목 설정 (Study Helper로 변경)
st.title("📚 Study Helper")
st.write("---")

# 'lang' 초기화
if 'lang' not in st.session_state:
    st.session_state.lang = 'english'  # 기본 언어를 영어로 설정합니다.

# 저작권 유의사항 경고 메시지 추가
st.warning("저작물을 불법 복제하여 게시하는 경우 당사는 책임지지 않으며, 저작권법에 유의하여 파일을 올려주세요.")

# 채팅 인터페이스 함수 정의
def chat_interface():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 채팅 기록 표시
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            with st.chat_message("user"):
                st.write(chat["message"])
        else:
            with st.chat_message("assistant"):
                st.write(chat["message"])

    # 채팅 입력
    if st.session_state.lang == 'korean':
        st.write("## ChatGPT와의 채팅")
        user_chat_input = st.chat_input("메시지를 입력하세요:")
    else:
        st.write("## Chat with ChatGPT")
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

def add_chat_message(role, message):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({"role": role, "message": message})

# PDF를 텍스트로 변환하는 함수
def pdf_to_text(upload_file):
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

# PPTX에서 텍스트 추출
def pptx_to_text(upload_file):
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

# 이미지에서 텍스트 추출 (한글+영어 지원)
def image_to_text(uploaded_image):
    try:
        image = Image.open(uploaded_image)
        text = pytesseract.image_to_string(image, lang='kor+eng')  # 한국어와 영어 지원
        return text
    except Exception as e:
        st.error(f'이미지에서 텍스트를 추출하는 중 오류가 발생했습니다: {e}')
        return ""

# HWP에서 텍스트 추출
def hwp_to_text(upload_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.hwp') as tmp:
            tmp.write(upload_file.getvalue())
            tmp_path = tmp.name
        # hwp5txt 툴 필요
        result = subprocess.run(["hwp5txt", tmp_path], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            st.error("HWP에서 텍스트를 추출할 수 없습니다. hwp5txt 툴이 설치되어 있는지 확인해주세요.")
            return ""
    except Exception as e:
        st.error(f"HWP 처리 중 오류가 발생했습니다: {e}")
        return ""

# 언어 감지 함수
def detect_language(text):
    try:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        prompt = f"다음 텍스트의 언어를 ISO 639-1 코드로 감지해 주세요 (예: 'en'은 영어, 'ko'는 한국어):\n\n{text[:500]}"
        messages = [HumanMessage(content=prompt)]
        response = llm(messages)
        language_code = response.content.strip().lower().split()[0]
        return language_code
    except Exception as e:
        st.error(f"언어 감지 중 오류가 발생했습니다: {e}")
        return "unknown"

# 요약 생성 함수
def summarize_pdf(text, language):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    if language == 'korean':
        prompt = f"다음 텍스트를 읽고 서론, 본론, 결론으로 구성된 자세한 요약을 작성해 주세요:\n\n{text}"
    else:
        prompt = f"Read the following text and write a detailed summary with introduction, main body, and conclusion:\n\n{text}"
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    return response.content.strip()

def extract_key_summary_words_with_sources(text, language):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
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
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    if language == 'korean':
        prompt = f"다음 요약에서 중요한 용어 5~10개를 추출하고, 각 용어 정의와 텍스트 내 페이지 정보를 제공:\n\n{summary_text}"
    else:
        prompt = f"From the following summary, extract 5-10 important terms, provide detailed definitions and their page references:\n\n{summary_text}"
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    return response.content.strip()

def generate_quiz(text, language):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
    if language == 'korean':
        prompt = f"다음 내용을 기반으로 객관식 5문제 생성(4지선다), 정답 표시:\n\n{text}"
    else:
        prompt = f"Based on the following content, create 5 multiple-choice questions (4 options) and indicate the correct answer:\n\n{text}"
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    return response.content

def generate_exam_questions(text, language):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
    if language == 'korean':
        prompt = f"다음 내용을 기반으로 중요한 시험 문제 5개를 제시:\n\n{text}"
    else:
        prompt = f"Based on the following content, provide 5 important exam questions:\n\n{text}"
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    return response.content

def generate_questions_for_user(text, language):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
    if language == 'korean':
        prompt = f"다음 내용을 기반으로 사용자가 깊이 생각할 수 있는 질문 3개 제시:\n\n{text}"
    else:
        prompt = f"Based on the following content, generate 3 thoughtful questions for deeper understanding:\n\n{text}"
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    questions = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
    return questions

def ask_gpt_question(question, language):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
    if language == 'korean':
        prompt = f"다음 질문에 답변: {question}"
    else:
        prompt = question
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    return response.content

# 결과를 PPTX로 생성하는 함수
def create_ppt_from_text(text, filename="summary_output.pptx"):
    prs = Presentation()
    # 기본 레이아웃 사용 (제목 + 본문)
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    slide.shapes.title.text = "Summary"
    slide.placeholders[1].text = text

    # 추가적인 슬라이드를 만들고 싶다면 필요에 따라 확장 가능
    buf = BytesIO()
    prs.save(buf)
    buf.seek(0)
    return buf

if "processed" not in st.session_state:
    st.session_state.processed = False

# 파일 업로드 (HWP 포함)
uploaded_file = st.file_uploader("파일을 올려주세요 (PDF, PPTX, PNG, JPG, JPEG, HWP 지원)", type=['pdf', 'pptx', 'png', 'jpg', 'jpeg', 'hwp'])

chat_interface()

if uploaded_file is not None:
    filename = uploaded_file.name
    extension = os.path.splitext(filename)[1].lower()

    if extension == ".pdf":
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()

        if ("uploaded_file_hash" not in st.session_state or
                st.session_state.uploaded_file_hash != file_hash):
            st.session_state.uploaded_file_hash = file_hash
            # 이전 결과 초기화
            st.session_state.extracted_text = ""
            st.session_state.summary = ""
            st.session_state.keywords = ""
            st.session_state.term_info = ""
            st.session_state.quiz = ""
            st.session_state.exam_questions = ""
            st.session_state.gpt_questions = []
            st.session_state.processed = False

        if not st.session_state.processed:
            extracted_text = pdf_to_text(uploaded_file)
            if not extracted_text.strip():
                st.error("PDF에서 텍스트를 추출할 수 없습니다.")
                summary = ""
                st.session_state.summary = summary
            else:
                st.success("PDF에서 텍스트 추출 완료.")
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

                with st.spinner("요약 생성 중..."):
                    summary = summarize_pdf(extracted_text, lang)
                    st.session_state.summary = summary

                with st.spinner("핵심 단어 추출 중..."):
                    key_summary_words = extract_key_summary_words_with_sources(extracted_text, lang)
                    st.session_state.keywords = key_summary_words

                with st.spinner("중요 단어 정보 추출 중..."):
                    term_info = extract_and_search_terms(summary, extracted_text, language=lang)
                    st.session_state.term_info = term_info

                with st.spinner("퀴즈 생성 중..."):
                    quiz = generate_quiz(extracted_text, lang)
                    st.session_state.quiz = quiz

                with st.spinner("시험 문제 생성 중..."):
                    exam_questions = generate_exam_questions(extracted_text, lang)
                    st.session_state.exam_questions = exam_questions

                with st.spinner("사용자용 질문 생성 중..."):
                    gpt_questions = generate_questions_for_user(extracted_text, lang)
                    st.session_state.gpt_questions = gpt_questions

                st.session_state.processed = True
        else:
            extracted_text = st.session_state.extracted_text
            summary = st.session_state.summary
            key_summary_words = st.session_state.keywords
            term_info = st.session_state.term_info
            quiz = st.session_state.quiz
            exam_questions = st.session_state.exam_questions

    elif extension == ".pptx":
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()

        if ("uploaded_file_hash" not in st.session_state or
                st.session_state.uploaded_file_hash != file_hash):
            st.session_state.uploaded_file_hash = file_hash
            st.session_state.extracted_text = ""
            st.session_state.summary = ""
            st.session_state.keywords = ""
            st.session_state.term_info = ""
            st.session_state.quiz = ""
            st.session_state.exam_questions = ""
            st.session_state.gpt_questions = []
            st.session_state.processed = False

        if not st.session_state.processed:
            extracted_text = pptx_to_text(uploaded_file)
            if not extracted_text.strip():
                st.error("PPTX에서 텍스트를 추출할 수 없습니다.")
                summary = ""
                st.session_state.summary = summary
            else:
                st.success("PPTX에서 텍스트 추출 완료.")
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

                with st.spinner("요약 생성 중..."):
                    summary = summarize_pdf(extracted_text, lang)
                    st.session_state.summary = summary

                with st.spinner("핵심 단어 추출 중..."):
                    key_summary_words = extract_key_summary_words_with_sources(extracted_text, lang)
                    st.session_state.keywords = key_summary_words

                with st.spinner("중요 단어 정보 추출 중..."):
                    term_info = extract_and_search_terms(summary, extracted_text, language=lang)
                    st.session_state.term_info = term_info

                with st.spinner("퀴즈 생성 중..."):
                    quiz = generate_quiz(extracted_text, lang)
                    st.session_state.quiz = quiz

                with st.spinner("시험 문제 생성 중..."):
                    exam_questions = generate_exam_questions(extracted_text, lang)
                    st.session_state.exam_questions = exam_questions

                with st.spinner("사용자용 질문 생성 중..."):
                    gpt_questions = generate_questions_for_user(extracted_text, lang)
                    st.session_state.gpt_questions = gpt_questions

                st.session_state.processed = True
            else:
                extracted_text = st.session_state.extracted_text
                summary = st.session_state.summary
                key_summary_words = st.session_state.keywords
                term_info = st.session_state.term_info
                quiz = st.session_state.quiz
                exam_questions = st.session_state.exam_questions

    elif extension in [".png", ".jpg", ".jpeg"]:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()

        if ("uploaded_file_hash" not in st.session_state or
                st.session_state.uploaded_file_hash != file_hash):
            st.session_state.uploaded_file_hash = file_hash
            st.session_state.extracted_text = ""
            st.session_state.summary = ""
            st.session_state.keywords = ""
            st.session_state.term_info = ""
            st.session_state.quiz = ""
            st.session_state.exam_questions = ""
            st.session_state.gpt_questions = []
            st.session_state.processed = False

        if not st.session_state.processed:
            extracted_text = image_to_text(uploaded_file)
            if not extracted_text.strip():
                st.error("이미지에서 텍스트를 추출할 수 없습니다.")
                summary = ""
                st.session_state.summary = summary
            else:
                st.success("이미지에서 텍스트 추출 완료.")
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

                with st.spinner("요약 생성 중..."):
                    summary = summarize_pdf(extracted_text, lang)
                    st.session_state.summary = summary

                with st.spinner("핵심 단어 추출 중..."):
                    key_summary_words = extract_key_summary_words_with_sources(extracted_text, lang)
                    st.session_state.keywords = key_summary_words

                with st.spinner("중요 단어 정보 추출 중..."):
                    term_info = extract_and_search_terms(summary, extracted_text, language=lang)
                    st.session_state.term_info = term_info

                with st.spinner("퀴즈 생성 중..."):
                    quiz = generate_quiz(extracted_text, lang)
                    st.session_state.quiz = quiz

                with st.spinner("시험 문제 생성 중..."):
                    exam_questions = generate_exam_questions(extracted_text, lang)
                    st.session_state.exam_questions = exam_questions

                with st.spinner("사용자용 질문 생성 중..."):
                    gpt_questions = generate_questions_for_user(extracted_text, lang)
                    st.session_state.gpt_questions = gpt_questions

                st.session_state.processed = True
            else:
                extracted_text = st.session_state.extracted_text
                summary = st.session_state.summary
                key_summary_words = st.session_state.keywords
                term_info = st.session_state.term_info
                quiz = st.session_state.quiz
                exam_questions = st.session_state.exam_questions

    elif extension == ".hwp":
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()

        if ("uploaded_file_hash" not in st.session_state or
                st.session_state.uploaded_file_hash != file_hash):
            st.session_state.uploaded_file_hash = file_hash
            st.session_state.extracted_text = ""
            st.session_state.summary = ""
            st.session_state.keywords = ""
            st.session_state.term_info = ""
            st.session_state.quiz = ""
            st.session_state.exam_questions = ""
            st.session_state.gpt_questions = []
            st.session_state.processed = False

        if not st.session_state.processed:
            extracted_text = hwp_to_text(uploaded_file)
            if not extracted_text.strip():
                st.error("HWP에서 텍스트를 추출할 수 없습니다.")
                summary = ""
                st.session_state.summary = summary
            else:
                st.success("HWP에서 텍스트 추출 완료.")
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

                with st.spinner("요약 생성 중..."):
                    summary = summarize_pdf(extracted_text, lang)
                    st.session_state.summary = summary

                with st.spinner("핵심 단어 추출 중..."):
                    key_summary_words = extract_key_summary_words_with_sources(extracted_text, lang)
                    st.session_state.keywords = key_summary_words

                with st.spinner("중요 단어 정보 추출 중..."):
                    term_info = extract_and_search_terms(summary, extracted_text, language=lang)
                    st.session_state.term_info = term_info

                with st.spinner("퀴즈 생성 중..."):
                    quiz = generate_quiz(extracted_text, lang)
                    st.session_state.quiz = quiz

                with st.spinner("시험 문제 생성 중..."):
                    exam_questions = generate_exam_questions(extracted_text, lang)
                    st.session_state.exam_questions = exam_questions

                with st.spinner("사용자용 질문 생성 중..."):
                    gpt_questions = generate_questions_for_user(extracted_text, lang)
                    st.session_state.gpt_questions = gpt_questions

                st.session_state.processed = True
            else:
                extracted_text = st.session_state.extracted_text
                summary = st.session_state.summary
                key_summary_words = st.session_state.keywords
                term_info = st.session_state.term_info
                quiz = st.session_state.quiz
                exam_questions = st.session_state.exam_questions

    else:
        st.error("지원하지 않는 파일 형식입니다. PDF, PPTX, PNG, JPG, JPEG, HWP 파일만 올려주세요.")

    # 결과 표시
    if st.session_state.get("processed", False):
        if 'summary' in st.session_state and st.session_state.summary.strip():
            st.write("## 요약 결과")
            st.write(st.session_state.summary)
        else:
            st.write("## 요약 결과를 표시할 수 없습니다.")

        if ('keywords' in st.session_state and
                st.session_state.keywords.strip()):
            st.write("## 핵심 요약 단어 및 출처")
            st.write(st.session_state.keywords)

        if ('term_info' in st.session_state and
                st.session_state.term_info.strip()):
            st.write("## 요약 내 중요한 단어 정보")
            st.write(st.session_state.term_info)

        if 'quiz' in st.session_state and st.session_state.quiz.strip():
            st.write("## 생성된 퀴즈")
            st.write(st.session_state.quiz)

        if ('exam_questions' in st.session_state and
                st.session_state.exam_questions.strip()):
            st.write("## 생성된 시험 문제")
            st.write(st.session_state.exam_questions)

        # PPT 생성 기능 추가
        st.write("---")
        if st.button("요약 내용을 PPT로 다운로드"):
            ppt_buffer = create_ppt_from_text(st.session_state.summary)
            st.download_button(
                label="PPT 다운로드",
                data=ppt_buffer,
                file_name="summary_output.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
            )

# 키워드 검색 기능
if st.session_state.get("processed", False):
    st.write("---")
    if st.session_state.lang == 'korean':
        st.write("## 🔍 키워드 검색")
        search_query = st.text_input("검색할 키워드를 입력하세요:")
    else:
        st.write("## 🔍 Keyword Search")
        search_query = st.text_input("Enter a keyword to search:")

    if search_query:
        search_results = []
        for line in st.session_state.extracted_text.split('\n'):
            if search_query.lower() in line.lower():
                search_results.append(line.strip())
        if search_results:
            if st.session_state.lang == 'korean':
                st.write("### 검색 결과:")
            else:
                st.write("### Search Results:")
            for result in search_results:
                st.write(f"- {result}")
        else:
            if st.session_state.lang == 'korean':
                st.write("검색 결과가 없습니다.")
            else:
                st.write("No results found.")

# GPT가 사용자에게 질문
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
                with st.spinner("GPT가 응답 검토 중..."):
                    if st.session_state.lang == 'korean':
                        feedback_prompt = f"{question}\n\n사용자 답변: {user_answer}\n\n피드백을 제공해 주세요."
                    else:
                        feedback_prompt = f"{question}\n\nUser's answer: {user_answer}\n\nPlease provide feedback."
                    feedback = ask_gpt_question(feedback_prompt, st.session_state.lang)
                    if st.session_state.lang == 'korean':
                        st.write("### GPT의 피드백")
                    else:
                        st.write("### GPT's Feedback")
                    st.write(feedback)

st.write("---")
st.info("**ChatGPT는 실수를 할 수 있습니다. 중요한 정보를 확인하세요.**")





