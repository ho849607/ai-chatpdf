import os
import tempfile
import streamlit as st
from pypdf import PdfReader  # pypdf2 사용
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API key 설정
openai_api_key = os.getenv("OPENAI_API_KEY")  # 환경 변수에서 API 키 가져오기

# 제목
st.title("ChatPDF")
st.write("---")

# 파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 올려주세요", type=['pdf'])
st.write("---")

# PDF to document conversion
def pdf_to_document(upload_file):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_filepath = os.path.join(temp_dir, upload_file.name)
        with open(temp_filepath, "wb") as f:
            f.write(upload_file.getvalue())
        # pypdf2로 PDF 읽기
        reader = PdfReader(temp_filepath)
        pages = [page.extract_text() for page in reader.pages if page.extract_text()]
    return pages

# Handle file upload
if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        # Load the PDF and split into pages
        pages = pdf_to_document(uploaded_file)
        
        # Text Splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=20,
            length_function=len
        )
        # 페이지별 텍스트 분할
        texts = text_splitter.create_documents(pages)
        
        # Embedding Model
        embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # Load into Chroma Vectorstore with local persistence
        persist_directory = "./chroma_db"  # 로컬 데이터베이스 저장 위치 지정
        db = Chroma.from_documents(texts, embeddings_model, persist_directory=persist_directory)
        
        # Create a RetrievalQA chain
        retriever = db.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=openai_api_key),
            chain_type="stuff", 
            retriever=retriever
        )

        # Question input
        st.header("PDF에게 질문해보세요.")
        question = st.text_input("질문을 입력하세요.")
        
        if st.button('질문하기'):
            # Process the question and get the answer
            answer = qa_chain.run(question)
            st.write("답변: ", answer)
    else:
        st.write("지원하지 않는 파일 형식입니다. PDF 파일만 올려주세요.")


