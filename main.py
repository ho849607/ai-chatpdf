import os
import sys
import streamlit as st
from io import BytesIO
from dotenv import load_dotenv
import pdfplumber  # PDF 파일에서 텍스트 추출
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch  # 인메모리 벡터스토어 사용
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import HumanMessage
import openai
from pathlib import Path

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

# PDF 요약 생성 함수
def generate_summary(extracted_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.create_documents([extracted_text])
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        max_tokens=1500,
        openai_api_key=openai_api_key
    )
    summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = summary_chain.run(texts)
    return summary

# GPT 질문 생성 함수
def generate_gpt_questions(extracted_text):
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=1500,
        openai_api_key=openai_api_key
    )
    prompt = f"다음 텍스트에서 중요한 개념이나 주제에 대해 사용자가 더 깊이 생각할 수 있도록 질문 5개를 만들어주세요:\n\n{extracted_text}"
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    return response.content

# 예상 시험 문제 생성 함수
def generate_exam_questions(extracted_text):
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.5,
        max_tokens=1500,
        openai_api_key=openai_api_key
    )
    prompt = f"다음 내용에 기반하여 예상되는 중요한 시험 문제 5개를 만들어주세요:\n\n{extracted_text}"
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    return response.content

# 주요 키워드 추출 함수
def extract_keywords(extracted_text):
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.5,
        max_tokens=1500,
        openai_api_key=openai_api_key
    )
    prompt = f"다음 텍스트에서 주요 키워드 10개를 추출해 주세요:\n\n{extracted_text}"
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    return response.content

# 파일 업로드 처리
if uploaded_file is not None and uploaded_file.type == "application/pdf":
    st.success("PDF 업로드 완료. 내용을 처리 중입니다...")

    # 백그라운드에서 즉시 처리 시작
    with st.spinner("PDF 내용을 처리하고 있습니다..."):
        extracted_text = pdf_to_text(uploaded_file)

        if extracted_text.strip():
            # 모든 프로세스를 병렬로 자동 실행
            summarize_text = generate_summary(extracted_text)
            gpt_questions = generate_gpt_questions(extracted_text)
            exam_questions = generate_exam_questions(extracted_text)
            keywords = extract_keywords(extracted_text)

            # 모든 결과를 한 번에 표시
            st.write("## 요약 결과")
            st.write(summarize_text)

            st.write("## GPT 질문")
            st.write(gpt_questions)

            st.write("## 예상 시험 문제")
            st.write(exam_questions)

            st.write("## 주요 키워드")
            st.write(keywords)
        else:
            st.error("PDF에서 텍스트를 추출할 수 없습니다. 다른 PDF를 시도해보세요.")
else:
    if uploaded_file is not None:
        st.error("지원하지 않는 파일 형식입니다. PDF 파일만 올려주세요.")



