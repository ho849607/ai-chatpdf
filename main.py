import os
import streamlit as st
from io import BytesIO
from dotenv import load_dotenv
import pdfplumber
import openai
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# ----------------------- NLTK 설정 시작 -----------------------

# NLTK 데이터 디렉토리 정의 및 경로 확인
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_dir)

# 현재 NLTK 데이터 경로 출력
st.write("NLTK 데이터 경로 확인:")
for path in nltk.data.path:
    st.write(path)

# 수동으로 punkt 다운로드한 경우 해당 경로 확인
try:
    nltk.data.find('tokenizers/punkt')
    st.write("punkt 데이터가 존재합니다.")
except LookupError:
    st.error("punkt 데이터를 찾을 수 없습니다. 다운로드를 시도하세요.")

# ----------------------- NLTK 설정 종료 -----------------------

# 환경 변수 로드
dotenv_path = Path('.env')  # Streamlit에서는 __file__을 사용할 수 없으므로 현재 디렉토리의 .env 파일을 참조
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

# 요약 생성 함수
def summarize_pdf(text):
    try:
        # 문장 단위로 텍스트를 분할
        sentences = sent_tokenize(text, language='english')

        # 청크 생성 (1000자당 200자 오버랩)
        chunk_size = 1000
        chunk_overlap = 200
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 > chunk_size:
                # 오버랩을 위해 마지막 200자를 유지
                overlap = current_chunk[-chunk_overlap:]
                chunks.append(current_chunk.strip())
                current_chunk = overlap + " " + sentence
            else:
                current_chunk += " " + sentence
        if current_chunk:
            chunks.append(current_chunk.strip())

        # 각 청크를 요약
        summaries = []
        for chunk in chunks:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=f"Summarize the following text:\n\n{chunk}",
                max_tokens=150,
                temperature=0.5,
            )
            summaries.append(response.choices[0].text.strip())

        # 요약된 청크들을 다시 한번 요약
        combined_summary = "\n".join(summaries)
        final_response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Summarize the following summaries into a coherent overall summary:\n\n{combined_summary}",
            max_tokens=150,
            temperature=0.5,
        )
        return final_response.choices[0].text.strip()
    except Exception as e:
        st.error(f"요약 생성 중 오류 발생: {e}")
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
            st.write("---")

            # 자동으로 요약 생성
            with st.spinner("요약을 생성하고 있습니다..."):
                summary = summarize_pdf(extracted_text)
                if summary:
                    st.write("## 요약 결과")
                    st.write(summary)

else:
    st.info("PDF 파일을 업로드해주세요.")


    
