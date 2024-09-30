__import__('pysqlite3')
import sys
sys.module['sqlite3']=sys.modules.pop('psyqlite3')


import os
import streamlit as st
from io import BytesIO
from dotenv import load_dotenv
import pdfplumber  # PDF 파일에서 텍스트 추출
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma  # langchain_community에서 임포트
from langchain_community.chat_models import ChatOpenAI  # langchain_community에서 임포트
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document  # Document 클래스 임포트
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain

# 환경 변수 로드
load_dotenv()

# API 키 설정
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API 키가 설정되지 않았습니다.")
    st.stop()

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
                        # 요약 생성
                        llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
                        summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
                        docs = [Document(page_content=extracted_text)]
                        summary = summary_chain.run(docs)
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
                        llm = OpenAI(temperature=0.5, openai_api_key=openai_api_key)
                        prompt = f"다음 내용에 기반하여 예상되는 중요한 시험 문제 5개를 만들어주세요:\n\n{extracted_text}"
                        questions = llm(prompt)
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
                        llm = OpenAI(temperature=0.5, openai_api_key=openai_api_key)
                        prompt = f"다음 내용에 기반하여 객관식 퀴즈 5개를 만들어주세요. 각 질문에는 4개의 선택지가 있어야 하며, 정답을 표시해주세요:\n\n{extracted_text}"
                        quiz = llm(prompt)
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
            # 텍스트 분할
            chunks = text_splitter.split_text(extracted_text)
            # 문서 생성
            texts = [Document(page_content=chunk) for chunk in chunks]
            
            if not texts:
                st.error("텍스트를 분할할 수 없습니다.")
            else:
                # 임베딩 모델
                embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
                
                # Chroma 벡터스토어에 로드
                persist_directory = "./chroma_db"
                collection_name = "pdf_collection"
                try:
                    db = Chroma.from_documents(
                        texts,
                        embedding=embeddings_model,
                        persist_directory=persist_directory,
                        collection_name=collection_name
                    )
                except Exception as e:
                    st.error(f"Chroma 벡터스토어 생성 중 오류가 발생했습니다: {e}")
                    st.stop()
                
                # RetrievalQA 체인 생성
                retriever = db.as_retriever()
                try:
                    llm = ChatOpenAI(
                        model_name="gpt-3.5-turbo",
                        temperature=0,
                        openai_api_key=openai_api_key
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
                        try:
                            answer = qa_chain.run(question)
                            st.write("### 답변")
                            st.write(answer)
                        except Exception as e:
                            st.error(f"질문 처리 중 오류가 발생했습니다: {e}")
    else:
        st.error("지원하지 않는 파일 형식입니다. PDF 파일만 올려주세요.")





