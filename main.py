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
from langdetect import detect

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
st.title("PDF 학습 도우미")
st.write("---")

# PDF를 텍스트로 변환하는 함수
def pdf_to_text(upload_file):
    try:
        with pdfplumber.open(BytesIO(upload_file.read())) as pdf:
            pages = [page.extract_text() for page in pdf.pages]
        return "\n".join(pages)
    except Exception as e:
        st.error(f"PDF에서 텍스트를 추출하는 중 오류가 발생했습니다: {e}")
        return ""

# 언어 감지 함수
def detect_language(text):
    try:
        return detect(text)
    except Exception as e:
        st.error(f"언어 감지 중 오류가 발생했습니다: {e}")
        return "unknown"

# 요약 생성 함수
def summarize_pdf(text, language):
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

            prompt = f"Provide a detailed definition and context for the term '{term}' in {language}."
            messages = [HumanMessage(content=prompt)]
            response = llm(messages)
            info = response.content
            term_info[term] = info
        except Exception as e:
            term_info[term] = f"Error retrieving information: {e}"
    return term_info

# 파일 업로드 및 데이터 처리
if "processed" not in st.session_state:
    st.session_state.processed = False

uploaded_file = st.file_uploader("PDF 파일을 올려주세요", type=['pdf'])

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        extracted_text = pdf_to_text(uploaded_file)

        if not extracted_text.strip():
            st.error("PDF에서 텍스트를 추출할 수 없습니다. 다른 PDF를 시도해보세요.")
        else:
            st.success("PDF에서 텍스트를 추출했습니다.")
            language = detect_language(extracted_text)
            lang = 'korean' if language == 'ko' else 'english'

            st.write(f"### 감지된 언어: {'한국어' if lang == 'korean' else '영어'}")

            # 요약 생성 및 저장
            with st.spinner("요약을 생성하고 있습니다..."):
                summary = summarize_pdf(extracted_text, lang)
                st.write("## 요약 결과")
                st.write(summary)
                st.session_state.summary = summary

            with st.spinner("요약 내 단어를 검색하고 있습니다..."):
                term_info = extract_and_search_terms(summary, extracted_text, language=lang)
                st.write("## 요약 내 중요한 단어 정보")
                for term, info in term_info.items():
                    st.write(f"### {term}")
                    st.write(info)

            st.session_state.processed = True
            st.session_state.lang = lang
            st.session_state.extracted_text = extracted_text
    else:
        st.error("지원하지 않는 파일 형식입니다. PDF 파일만 올려주세요.")

if st.session_state.processed:
    st.write("---")
    user_question = st.text_input(f"{'질문을 입력하세요' if st.session_state.lang == 'korean' else 'Enter your question'}:")
    if user_question:
        with st.spinner(f"{'GPT가 답변 중입니다...' if st.session_state.lang == 'korean' else 'GPT is responding...'}"):
            gpt_response = ask_gpt_question(user_question, st.session_state.lang)
            st.write(f"### {'GPT의 답변' if st.session_state.lang == 'korean' else 'GPT\'s Response'}")
            st.write(gpt_response)

