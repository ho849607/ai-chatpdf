import os
import streamlit as st
from io import BytesIO
from dotenv import load_dotenv
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import HumanMessage
import openai
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# NLTK 데이터 다운로드
nltk.download('punkt')
nltk.download('stopwords')



# 나머지 코드...

# 단어 추출 및 검색 함수
def extract_and_search_terms(summary_text):
    tokens = word_tokenize(summary_text, language='english')
    stop_words = set(stopwords.words('english'))
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
            prompt = f"Provide the definition and related information for the term '{term}'."
            messages = [HumanMessage(content=prompt)]
            response = llm(messages)
            info = response.content
            term_info[term] = info
        except Exception as e:
            term_info[term] = f"Error retrieving information: {e}"
    return term_info

# NLTK 데이터 다운로드 (최초 실행 시)
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.mkdir(nltk_data_dir)

nltk.data.path.append(nltk_data_dir)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_dir)

# 환경 변수 로드
dotenv_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

# API 키 설정
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    # Streamlit 사이드바에서 API 키 입력받기
    st.sidebar.title("API 설정")
    openai_api_key = st.sidebar.text_input("OpenAI API 키를 입력하세요.", type="password")
    if not openai_api_key:
        st.error("OpenAI API 키가 설정되지 않았습니다.")
        st.stop()

# OpenAI API 키 설정
openai.api_key = openai_api_key

# 제목
st.title("PDF 학습 도우미")
st.write("---")

# 파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 올려주세요", type=['pdf'])
st.write("---")

# PDF를 텍스트로 변환하는 함수
def pdf_to_text(upload_file):
    try:
        with pdfplumber.open(BytesIO(upload_file.read())) as pdf:
            pages = [page.extract_text() for page in pdf.pages]
        text = "\n".join(pages)
        return text
    except Exception as e:
        st.error(f"PDF에서 텍스트를 추출하는 중 오류가 발생했습니다: {e}")
        return ""

# 단어 추출 및 검색 함수
def extract_and_search_terms(summary_text):
    # 언어를 명시적으로 지정하여 토큰화
    tokens = word_tokenize(summary_text, language='english')
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if w.isalnum() and w.lower() not in stop_words]
    # 빈도수 계산
    freq_dist = nltk.FreqDist(filtered_tokens)
    # 상위 5개 단어 선택
    important_terms = [word for word, freq in freq_dist.most_common(5)]
    # 각 단어에 대한 정보 검색
    term_info = {}
    for term in important_terms:
        try:
            # OpenAI를 사용하여 해당 단어의 정의 및 관련 정보 가져오기
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0,
                max_tokens=150,
                openai_api_key=openai_api_key
            )
            prompt = f"Provide the definition and related information for the term '{term}'."
            messages = [HumanMessage(content=prompt)]
            response = llm(messages)
            info = response.content
            term_info[term] = info
        except Exception as e:
            term_info[term] = f"Error retrieving information: {e}"
    return term_info

# 요약 생성 함수
def summarize_pdf(text):
    # 텍스트 분할기 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts]

    # 요약 생성
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        max_tokens=1500,
        openai_api_key=openai_api_key
    )
    summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = summary_chain({"input_documents": docs}, return_only_outputs=True)
    return summary['output_text']

# 예상 시험 문제 생성 함수
def generate_exam_questions(text):
    # 시험 문제 생성 프롬프트
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.5,
        max_tokens=1500,
        openai_api_key=openai_api_key
    )
    prompt = f"Based on the following content, create 5 important exam questions:\n\n{text}"
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    questions = response.content
    return questions

# 퀴즈 생성 함수
def generate_quiz(text):
    # 퀴즈 생성 프롬프트
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.5,
        max_tokens=1500,
        openai_api_key=openai_api_key
    )
    prompt = f"Based on the following content, create 5 multiple-choice quiz questions. Each question should have 4 options and indicate the correct answer:\n\n{text}"
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    quiz = response.content
    return quiz

# GPT가 질문 생성 함수
def generate_gpt_questions(text):
    # 질문 생성
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=1500,
        openai_api_key=openai_api_key
    )
    prompt = f"From the following text, create 5 questions that encourage deeper thinking about important concepts or topics:\n\n{text}"
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    gpt_questions = response.content
    return gpt_questions

# 파일 업로드 처리
if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        # PDF에서 텍스트 추출
        extracted_text = pdf_to_text(uploaded_file)

        if not extracted_text.strip():
            st.error("PDF에서 텍스트를 추출할 수 없습니다. 다른 PDF를 시도해보세요.")
        else:
            st.success("PDF에서 텍스트를 추출했습니다.")
            st.write("---")

            # 자동으로 요약, 시험 문제, 퀴즈, GPT 질문 생성
            with st.spinner("요약을 생성하고 있습니다..."):
                summary = summarize_pdf(extracted_text)
                st.write("## 요약 결과")
                st.write(summary)

            with st.spinner("요약 내 단어를 검색하고 있습니다..."):
                term_info = extract_and_search_terms(summary)
                st.write("## 요약 내 중요한 단어 정보")
                for term, info in term_info.items():
                    st.write(f"### {term}")
                    st.write(info)

            with st.spinner("시험 문제를 생성하고 있습니다..."):
                questions = generate_exam_questions(extracted_text)
                st.write("## 예상 시험 문제")
                st.write(questions)

            with st.spinner("퀴즈를 생성하고 있습니다..."):
                quiz = generate_quiz(extracted_text)
                st.write("## 생성된 퀴즈")
                st.write(quiz)

            with st.spinner("GPT가 질문을 생성하고 있습니다..."):
                gpt_questions = generate_gpt_questions(extracted_text)
                st.write("## GPT가 생성한 질문")
                st.write(gpt_questions)

    else:
        st.error("지원하지 않는 파일 형식입니다. PDF 파일만 올려주세요.")



