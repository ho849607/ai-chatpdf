# 필요한 라이브러리 임포트
import os
import streamlit as st
from io import BytesIO
from dotenv import load_dotenv
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document, HumanMessage
from langchain.chains.summarize import load_summarize_chain
import openai
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from langdetect import detect

# 추가된 라이브러리
from docx import Document as DocxDocument
from pptx import Presentation
import tempfile

# 초기 설정
nltk.download('punkt')
nltk.download('stopwords')

# .env 파일에서 환경 변수 로드
dotenv_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

# API 키 설정
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    openai_api_key = st.sidebar.text_input("OpenAI API 키를 입력하세요.", type="password")
    if not openai_api_key:
        st.error("OpenAI API 키가 설정되지 않았습니다.")
        st.stop()

openai.api_key = openai_api_key

# 제목 설정
st.title("학습 도우미")
st.write("---")

# 저작권 유의사항 경고 메시지 추가
st.warning("저작물을 불법 복제하여 게시하는 경우 당사는 책임지지 않으며, 저작권법에 유의하여 파일을 올려주세요.")

# 텍스트 추출 함수들
def pdf_to_text(upload_file):
    try:
        with pdfplumber.open(BytesIO(upload_file.read())) as pdf:
            pages = [page.extract_text() for page in pdf.pages]
        return "\n".join(pages)
    except Exception as e:
        st.error(f"PDF에서 텍스트를 추출하는 중 오류가 발생했습니다: {e}")
        return ""

def docx_to_text(upload_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_docx:
            temp_docx.write(upload_file.getbuffer())
            temp_docx_path = temp_docx.name
        doc = DocxDocument(temp_docx_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        os.remove(temp_docx_path)
        return '\n'.join(full_text)
    except Exception as e:
        st.error(f"DOCX에서 텍스트를 추출하는 중 오류가 발생했습니다: {e}")
        return ""

def pptx_to_text(upload_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pptx") as temp_pptx:
            temp_pptx.write(upload_file.getbuffer())
            temp_pptx_path = temp_pptx.name
        prs = Presentation(temp_pptx_path)
        text_runs = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text_runs.append(shape.text)
        os.remove(temp_pptx_path)
        return '\n'.join(text_runs)
    except Exception as e:
        st.error(f"PPTX에서 텍스트를 추출하는 중 오류가 발생했습니다: {e}")
        return ""

# 언어 감지 함수
def detect_language(text):
    try:
        return detect(text)
    except Exception as e:
        st.error(f"언어 감지 중 오류가 발생했습니다: {e}")
        return "unknown"

# 요약 생성 함수
def summarize_text(text, language):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts]
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=1500, openai_api_key=openai_api_key)
    summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
    return summary_chain({"input_documents": docs}, return_only_outputs=True)['output_text']

# 단어 추출 및 검색 함수
def extract_and_search_terms(summary_text, extracted_text, language='english'):
    tokens = word_tokenize(summary_text, language='english') if language == 'english' else summary_text.split()
    stop_words = set(stopwords.words('english')) if language == 'english' else []
    filtered_tokens = [w for w in tokens if w.isalnum() and w.lower() not in stop_words]

    freq_dist = nltk.FreqDist(filtered_tokens)
    important_terms = [word for word, freq in freq_dist.most_common(5)]

    term_info = {}
    for term in important_terms:
        try:
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0,
                max_tokens=150,
                openai_api_key=openai_api_key
            )

            prompt = (
                f"Provide a detailed definition and context for the term '{term}' in {language}."
                if language == 'english' else
                f"용어 '{term}'에 대한 자세한 정의와 맥락을 {language}로 제공해 주세요."
            )
            messages = [HumanMessage(content=prompt)]
            response = llm(messages)
            info = response.content
            term_info[term] = info
        except Exception as e:
            term_info[term] = f"정보를 가져오는 중 오류 발생: {e}"
    return term_info

# 퀴즈 생성 함수
def generate_quiz(text, language):
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.5,
        max_tokens=1500,
        openai_api_key=openai_api_key
    )
    prompt = (
        f"Based on the following content, create 5 multiple-choice quiz questions. Each question should have 4 options and indicate the correct answer:\n\n{text}"
        if language == 'english' else
        f"다음 내용을 기반으로 5개의 객관식 퀴즈 문제를 만들어 주세요. 각 질문은 4개의 선택지를 포함하고 정답을 표시해 주세요:\n\n{text}"
    )
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    return response.content

# 시험 문제 생성 함수
def generate_exam_questions(text, language):
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.5,
        max_tokens=1500,
        openai_api_key=openai_api_key
    )
    prompt = (
        f"Based on the following content, create 5 important exam questions:\n\n{text}"
        if language == 'english' else
        f"다음 내용을 기반으로 중요한 시험 문제 5개를 만들어 주세요:\n\n{text}"
    )
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    return response.content

# 사용자 질문에 대한 GPT 응답 함수
def ask_gpt_question(question, language):
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.5,
        max_tokens=1500,
        openai_api_key=openai_api_key
    )
    prompt = question if language == 'english' else f"다음 질문에 답해 주세요: {question}"
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    return response.content

# 파일 업로드 및 데이터 처리
if "processed" not in st.session_state:
    st.session_state.processed = False

uploaded_file = st.file_uploader("파일을 업로드하세요 (PDF, DOCX, PPTX)", type=['pdf', 'docx', 'pptx'])

if uploaded_file is not None:
    file_type = uploaded_file.type
    if file_type == "application/pdf":
        extracted_text = pdf_to_text(uploaded_file)
    elif file_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
        extracted_text = docx_to_text(uploaded_file)
    elif file_type == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
        extracted_text = pptx_to_text(uploaded_file)
    else:
        st.error("지원하지 않는 파일 형식입니다. PDF, DOCX, PPTX 파일만 올려주세요.")
        st.stop()

    if not extracted_text.strip():
        st.error("파일에서 텍스트를 추출할 수 없습니다. 다른 파일을 시도해보세요.")
    else:
        st.success("파일에서 텍스트를 추출했습니다.")
        language = detect_language(extracted_text)
        lang = 'korean' if language == 'ko' else 'english'

        st.write(f"### 감지된 언어: {'한국어' if lang == 'korean' else '영어'}")

        # 요약 생성 및 저장
        with st.spinner("요약을 생성하고 있습니다..."):
            summary = summarize_text(extracted_text, lang)
            st.write("## 요약 결과")
            st.write(summary)
            st.session_state.summary = summary

        # 중요 단어 정보 추출
        with st.spinner("요약 내 단어를 검색하고 있습니다..."):
            term_info = extract_and_search_terms(summary, extracted_text, language=lang)
            st.write("## 요약 내 중요한 단어 정보")
            for term, info in term_info.items():
                st.write(f"### {term}")
                st.write(info)

        # 퀴즈 생성
        with st.spinner("퀴즈를 생성하고 있습니다..."):
            quiz = generate_quiz(extracted_text, lang)
            st.write("## 생성된 퀴즈")
            st.write(quiz)

        # 시험 문제 생성
        with st.spinner("시험 문제를 생성하고 있습니다..."):
            exam_questions = generate_exam_questions(extracted_text, lang)
            st.write("## 생성된 시험 문제")
            st.write(exam_questions)

        st.session_state.processed = True
        st.session_state.lang = lang
        st.session_state.extracted_text = extracted_text

if st.session_state.processed:
    st.write("---")
    user_question = st.text_input(f"{'질문을 입력하세요' if st.session_state.lang == 'korean' else 'Enter your question'}:")
    if user_question:
        with st.spinner(f"{'GPT가 답변 중입니다...' if st.session_state.lang == 'korean' else 'GPT is responding...'}"):
            gpt_response = ask_gpt_question(user_question, st.session_state.lang)
            st.write(f"### {'GPT의 답변' if st.session_state.lang == 'korean' else 'GPT\'s Response'}")
            st.write(gpt_response)


