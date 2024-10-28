# 필요한 라이브러리 임포트
import os
import streamlit as st
from io import BytesIO
from dotenv import load_dotenv
import pdfplumber
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import openai
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# 커스텀 한국어 불용어 리스트 정의
korean_stopwords = ['이', '그', '저', '것', '수', '등', '들', '및', '더', '로', '를', '에', '의', '은', '는', '이', '가', '와', '과', '하다', '있다', '되다', '이다']

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

# 저작권 유의사항 경고 메시지 추가
st.warning("저작물을 불법 복제하여 게시하는 경우 당사는 책임지지 않으며, 저작권법에 유의하여 파일을 올려주세요.")

# PDF를 텍스트로 변환하는 함수
def pdf_to_text(upload_file):
    try:
        with pdfplumber.open(BytesIO(upload_file.read())) as pdf:
            pages = [page.extract_text() for page in pdf.pages]
        return "\n".join(pages)
    except Exception as e:
        st.error(f"PDF에서 텍스트를 추출하는 중 오류가 발생했습니다: {e}")
        return ""

# 언어 감지 함수 (OpenAI API 활용)
def detect_language(text):
    try:
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0
        )
        prompt = f"다음 텍스트의 언어를 ISO 639-1 코드로 감지해 주세요 (예: 'en'은 영어, 'ko'는 한국어):\n\n{text[:500]}"
        messages = [HumanMessage(content=prompt)]
        response = llm(messages)
        language_code = response.content.strip().lower()
        language_code = language_code.split()[0]
        return language_code
    except Exception as e:
        st.error(f"언어 감지 중 오류가 발생했습니다: {e}")
        return "unknown"

# 요약 생성 함수 (서론, 본론, 결론 구조로)
def summarize_pdf(text, language):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    if language == 'korean':
        prompt = f"다음 텍스트를 읽고 서론, 본론, 결론으로 구성된 자세한 요약을 작성해 주세요:\n\n{text}"
    else:
        prompt = f"Read the following text and write a detailed summary structured with an introduction, main body, and conclusion:\n\n{text}"
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    return response.content.strip()

# 핵심 요약 단어 추출 함수
def extract_key_summary_words(summary_text, language):
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0
    )
    if language == 'korean':
        prompt = f"다음 요약에서 중요한 키워드 5~10개를 추출해 주세요:\n\n{summary_text}"
    else:
        prompt = f"Extract 5 to 10 important keywords from the following summary:\n\n{summary_text}"
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    keywords = response.content.strip()
    return keywords

# 단어 추출 및 검색 함수
def extract_and_search_terms(summary_text, extracted_text, language='english'):
    if language == 'korean':
        tokens = summary_text.split()
        stop_words = set(korean_stopwords)
    else:
        tokens = word_tokenize(summary_text)
        stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if w.isalnum() and w.lower() not in stop_words]

    freq_dist = nltk.FreqDist(filtered_tokens)
    important_terms = [word for word, freq in freq_dist.most_common(5)]

    term_info = {}
    for term in important_terms:
        try:
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0
            )
            if language == 'korean':
                prompt = f"용어 '{term}'에 대한 자세한 정의와 맥락을 제공해 주세요."
            else:
                prompt = f"Provide a detailed definition and context for the term '{term}' in English."
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
        temperature=0.5
    )
    if language == 'korean':
        prompt = f"다음 내용을 기반으로 5개의 객관식 퀴즈 문제를 만들어 주세요. 각 질문은 4개의 선택지를 포함하고 정답을 표시해 주세요:\n\n{text}"
    else:
        prompt = f"Based on the following content, create 5 multiple-choice quiz questions. Each question should have 4 options and indicate the correct answer:\n\n{text}"
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    return response.content

# 시험 문제 생성 함수
def generate_exam_questions(text, language):
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.5
    )
    if language == 'korean':
        prompt = f"다음 내용을 기반으로 중요한 시험 문제 5개를 만들어 주세요:\n\n{text}"
    else:
        prompt = f"Based on the following content, create 5 important exam questions:\n\n{text}"
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    return response.content

# GPT가 사용자에게 질문을 생성하는 함수
def generate_questions_for_user(text, language):
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.5
    )
    if language == 'korean':
        prompt = f"다음 내용을 기반으로 사용자가 더 깊이 생각할 수 있도록 3개의 질문을 만들어 주세요:\n\n{text}"
    else:
        prompt = f"Based on the following content, generate 3 thoughtful questions to ask the user for a deeper understanding:\n\n{text}"
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    questions = response.content.strip().split('\n')
    return questions

# 사용자 질문에 대한 GPT 응답 함수
def ask_gpt_question(question, language):
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.5
    )
    if language == 'korean':
        prompt = f"다음 질문에 답변해 주세요: {question}"
    else:
        prompt = question
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    return response.content

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
            language_code = detect_language(extracted_text)
            if language_code == 'ko':
                lang = 'korean'
                language_name = '한국어'
            elif language_code == 'en':
                lang = 'english'
                language_name = '영어'
            else:
                lang = 'english'  # 기본값을 영어로 설정
                language_name = '알 수 없음 (영어로 진행합니다)'

            st.write(f"### 감지된 언어: {language_name}")

            # 요약 생성 및 저장
            with st.spinner("요약을 생성하고 있습니다..."):
                summary = summarize_pdf(extracted_text, lang)
                st.write("## 요약 결과")
                st.write(summary)
                st.session_state.summary = summary

            # 핵심 요약 단어 추출
            with st.spinner("핵심 요약 단어를 추출하고 있습니다..."):
                key_summary_words = extract_key_summary_words(summary, lang)
                st.write("## 핵심 요약 단어")
                st.write(key_summary_words)
                st.session_state.keywords = key_summary_words

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

            # GPT가 사용자에게 질문 생성
            with st.spinner("GPT가 질문을 생성하고 있습니다..."):
                gpt_questions = generate_questions_for_user(extracted_text, lang)
                st.session_state.gpt_questions = gpt_questions

            st.session_state.processed = True
            st.session_state.lang = lang
            st.session_state.extracted_text = extracted_text
    else:
        st.error("지원하지 않는 파일 형식입니다. PDF 파일만 올려주세요.")

if st.session_state.get("processed", False):
    st.write("---")
    if st.session_state.lang == 'korean':
        user_question = st.text_input("질문을 입력하세요:")
    else:
        user_question = st.text_input("Enter your question:")
    if user_question:
        with st.spinner("GPT가 답변 중입니다..."):
            gpt_response = ask_gpt_question(user_question, st.session_state.lang)
            if st.session_state.lang == 'korean':
                st.write("### GPT의 답변")
            else:
                st.write("### GPT's Response")
            st.write(gpt_response)

    # GPT가 사용자에게 질문하고 사용자 응답 받기
    st.write("---")
    if st.session_state.lang == 'korean':
        st.write("## GPT가 당신에게 질문합니다")
    else:
        st.write("## GPT has questions for you")
    if "gpt_questions" in st.session_state:
        for idx, question in enumerate(st.session_state.gpt_questions):
            user_answer = st.text_input(f"**{question}**", key=f"gpt_question_{idx}")
            if user_answer:
                with st.spinner("GPT가 응답을 검토 중입니다..."):
                    if st.session_state.lang == 'korean':
                        feedback_prompt = f"{question}\n\n사용자의 답변: {user_answer}\n\n이에 대해 피드백을 제공해 주세요."
                    else:
                        feedback_prompt = f"{question}\n\nUser's answer: {user_answer}\n\nPlease provide feedback on this."
                    feedback = ask_gpt_question(feedback_prompt, st.session_state.lang)
                    if st.session_state.lang == 'korean':
                        st.write("### GPT의 피드백")
                    else:
                        st.write("### GPT's Feedback")
                    st.write(feedback)
