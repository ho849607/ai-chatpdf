__import__('pysqlite3')
import sys
sys.modules['sqlite3']=sys.module.pop('pysqlite3')
import os
import tempfile
import streamlit as st
from io import BytesIO
from dotenv import load_dotenv
import pdfplumber  # PDF 파일에서 텍스트 추출
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings  # 수정: langchain.embeddings로 변경
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document  # Document 클래스 임포트
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import HumanMessage  # HumanMessage 임포트
import openai  # OpenAI 패키지 임포트
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

# 파일 업로드 처리
if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        # PDF에서 텍스트 추출
        extracted_text = pdf_to_text(uploaded_file)

        if not extracted_text.strip():
            st.error("PDF에서 텍스트를 추출할 수 없습니다. 다른 PDF를 시도해보세요.")
        else:
            st.success("PDF에서 텍스트를 추출했습니다.")

            # 요약 섹션
            st.header("1. PDF 요약")
            if st.button("PDF 요약 생성"):
                with st.spinner("요약을 생성하고 있습니다..."):
                    try:
                        # 텍스트 분할기 설정
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200
                        )
                        texts = text_splitter.create_documents([extracted_text])

                        # 요약 생성
                        llm = ChatOpenAI(
                            model_name="gpt-3.5-turbo",
                            temperature=0,
                            max_tokens=1500,
                            openai_api_key=openai_api_key  # 수정: API 키 전달
                        )
                        summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
                        summary = summary_chain.run(texts)
                        st.write("## 요약 결과")
                        st.write(summary)
                    except Exception as e:
                        st.error(f"요약 생성 중 오류가 발생했습니다: {e}")

            st.write("---")

            # 예상 시험 문제 섹션
            st.header("2. 예상 시험 문제 생성")
            if st.button("예상 시험 문제 생성"):
                with st.spinner("시험 문제를 생성하고 있습니다..."):
                    try:
                        # 시험 문제 생성 프롬프트
                        llm = ChatOpenAI(
                            model_name="gpt-3.5-turbo",
                            temperature=0.5,
                            max_tokens=1500,
                            openai_api_key=openai_api_key  # 수정: API 키 전달
                        )
                        prompt = f"다음 내용에 기반하여 예상되는 중요한 시험 문제 5개를 만들어주세요:\n\n{extracted_text}"
                        messages = [HumanMessage(content=prompt)]
                        response = llm(messages)
                        questions = response.content
                        st.write("## 예상 시험 문제")
                        st.write(questions)
                    except Exception as e:
                        st.error(f"시험 문제 생성 중 오류가 발생했습니다: {e}")

            st.write("---")

            # 퀴즈 섹션
            st.header("3. 학습 퀴즈")
            if st.button("학습 퀴즈 생성"):
                with st.spinner("퀴즈를 생성하고 있습니다..."):
                    try:
                        # 퀴즈 생성 프롬프트
                        llm = ChatOpenAI(
                            model_name="gpt-3.5-turbo",
                            temperature=0.5,
                            max_tokens=1500,
                            openai_api_key=openai_api_key  # 수정: API 키 전달
                        )
                        prompt = f"다음 내용에 기반하여 객관식 퀴즈 5개를 만들어주세요. 각 질문에는 4개의 선택지가 있어야 하며, 정답을 표시해주세요:\n\n{extracted_text}"
                        messages = [HumanMessage(content=prompt)]
                        response = llm(messages)
                        quiz = response.content
                        st.write("## 생성된 퀴즈")
                        st.write(quiz)
                    except Exception as e:
                        st.error(f"퀴즈 생성 중 오류가 발생했습니다: {e}")

            st.write("---")

            # PDF 내용에 대한 질문 섹션
            st.header("4. PDF 내용 질문하기")
            # 텍스트 분할기
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )
            texts = text_splitter.create_documents([extracted_text])

            if not texts:
                st.error("텍스트를 분할할 수 없습니다.")
            else:
                # 임베딩 모델
                embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)  # 수정: API 키 전달

                # Chroma 벡터스토어에 로드
                persist_directory = "./chroma_db"
                collection_name = "pdf_collection"
                try:
                    # Chroma 인스턴스 생성
                    db = Chroma(
                        collection_name=collection_name,
                        persist_directory=persist_directory,
                        embedding_function=embeddings_model,
                    )
                    # 문서 추가
                    db.add_documents(texts)
                except Exception as e:
                    st.error(f"Chroma 벡터스토어 생성 중 오류가 발생했습니다: {e}")
                    st.stop()

                # RetrievalQA 체인 생성
                retriever = db.as_retriever()
                try:
                    llm = ChatOpenAI(
                        model_name="gpt-3.5-turbo",
                        temperature=0,
                        max_tokens=1500,
                        openai_api_key=openai_api_key  # 수정: API 키 전달
                    )
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        retriever=retriever,
                        chain_type="stuff"
                    )
                except Exception as e:
                    st.error(f"RetrievalQA 체인 생성 중 오류가 발생했습니다: {e}")
                    st.stop()

                # 질문 입력
                question = st.text_input("궁금한 점을 질문하세요.", key="qa_question")

                if st.button('질문하기'):
                    if question.strip() == "":
                        st.error("질문을 입력해주세요.")
                    else:
                        # 질문 처리 및 답변 생성
                        with st.spinner("답변을 생성하고 있습니다..."):
                            try:
                                answer = qa_chain.run(question)
                                st.write("### 답변")
                                st.write(answer)
                            except Exception as e:
                                st.error(f"질문 처리 중 오류가 발생했습니다: {e}")

            st.write("---")

            # GPT가 사용자에게 질문하는 섹션
            st.header("5. GPT가 묻는 질문에 답변해보세요")

            if st.button("GPT 질문 생성"):
                with st.spinner("GPT가 질문을 생성하고 있습니다..."):
                    try:
                        # 질문 생성
                        llm = ChatOpenAI(
                            model_name="gpt-3.5-turbo",
                            temperature=0.7,
                            max_tokens=1500,
                            openai_api_key=openai_api_key  # 수정: API 키 전달
                        )
                        prompt = f"다음 텍스트에서 중요한 개념이나 주제에 대해 사용자가 더 깊이 생각할 수 있도록 질문 5개를 만들어주세요:\n\n{extracted_text}"
                        messages = [HumanMessage(content=prompt)]
                        response = llm(messages)
                        gpt_questions = response.content
                        st.session_state['gpt_questions'] = gpt_questions  # 세션에 저장
                        st.write("## GPT가 생성한 질문")
                        st.write(gpt_questions)
                    except Exception as e:
                        st.error(f"질문 생성 중 오류가 발생했습니다: {e}")

            # GPT의 질문이 생성되었을 때만 표시
            if 'gpt_questions' in st.session_state:
                st.write("---")
                st.write("### GPT의 질문에 답변해보세요.")
                user_response = st.text_area("질문에 대한 답변을 입력하세요.", key="user_response")

                if st.button("답변 제출"):
                    if user_response.strip() == "":
                        st.error("답변을 입력해주세요.")
                    else:
                        with st.spinner("GPT가 답변을 검토하고 있습니다..."):
                            try:
                                # 피드백 생성
                                llm = ChatOpenAI(
                                    model_name="gpt-3.5-turbo",
                                    temperature=0,
                                    max_tokens=4960,
                                    openai_api_key=openai_api_key  # 수정: API 키 전달
                                )
                                feedback_prompt = f"사용자의 답변: {user_response}\n이 답변에 대해 친절하고 건설적인 피드백을 3~5문장으로 제공해주세요."
                                messages = [HumanMessage(content=feedback_prompt)]
                                response = llm(messages)
                                feedback = response.content
                                st.write("## GPT의 피드백")
                                st.write(feedback)
                            except Exception as e:
                                st.error(f"피드백 생성 중 오류가 발생했습니다: {e}")

    else:
        st.error("지원하지 않는 파일 형식입니다. PDF 파일만 올려주세요.")








