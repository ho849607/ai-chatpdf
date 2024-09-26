__import__('pysqlite3')
import sys
sys.modules['sqlite3']=sys.modules.pop('pysqlite3')

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os

# 제목
st.title("ChatPDF")
st.write("---")

# 파일 업로드
uploaded_file = st.file_uploader("PDF파일을 올려주세요",type=['pdf','jpg'])
st.write("---")

# PDF to document conversion
def pdf_to_document(upload_file):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_filepath = os.path.join(temp_dir, upload_file.name)
        with open(temp_filepath, "wb") as f:
            f.write(upload_file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        pages = loader.load_and_split()
    return pages

# Handle file upload
if uploaded_file is not None:
    # Load the PDF and split into pages
    pages = pdf_to_document(uploaded_file)
    
    # Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=20,
        length_function=len
    )
    texts = text_splitter.split_documents(pages)
    
    # Embedding Model
    embeddings_model = OpenAIEmbeddings()
    
    # Load into Chroma Vectorstore with local persistence
    persist_directory = "./chroma_db"  # 로컬 데이터베이스 저장 위치 지정
    db = Chroma.from_documents(texts, embeddings_model, persist_directory=persist_directory)
    
    # Create a RetrievalQA chain
    retriever = db.as_retriever()  # 괄호 추가
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4", temperature=0),  # 모델 이름 수정
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
