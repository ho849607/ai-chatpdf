import os
import streamlit as st
from io import BytesIO
from dotenv import load_dotenv
import pdfplumber
from pptx import Presentation
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.callbacks import StreamingStdOutCallbackHandler
import openai
from pathlib import Path
import hashlib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from PIL import Image
import pytesseract
import subprocess
import tempfile
import docx  # python-docx

############################
# 초기 환경 설정
############################

# Tesseract 경로 (사용 환경에 맞춰 수정하세요)
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\username\AppData\Local\Tesseract-OCR\tesseract.exe"

# NLTK 리소스 다운로드
nltk.download('punkt')
nltk.download('stopwords')

# 한국어 불용어 리스트 (필요 시 수정/추가)
korean_stopwords = [
    '이', '그', '저', '것', '수', '등', '들', '및', '더', '로', '를', '에',
    '의', '은', '는', '가', '와', '과', '하다', '있다', '되다', '이다',
    '으로', '에서', '까지', '부터', '만', '그리고', '하지만', '그러나'
]

# .env 파일에서 API 키 읽기
dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    openai_api_key = st.sidebar.text_input("OpenAI API 키를 입력하세요.", type="password")
    if not openai_api_key:
        st.error("OpenAI API 키가 설정되지 않았습니다.")
        st.stop()

openai.api_key = openai_api_key

############################
# Streamlit UI
############################

st.title("📚 Study Helper (GPT-4)")
st.write("---")

if 'lang' not in st.session_state:
    st.session_state.lang = 'english'

st.warning("저작물을 불법 복제·게시하면 책임지지 않으며, 저작권법에 유의해주세요.")

############################
# 주요 함수
############################

def add_chat_message(role, message):
    """채팅 메시지를 세션에 기록"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({"role": role, "message": message})

def ask_gpt_question(question, language):
    """ChatOpenAI(GPT-4)로 질문"""
    llm = ChatOpenAI(
        model_name="gpt-4", 
        temperature=0, 
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    if language == 'korean':
        prompt = f"다음 질문에 답변해 주세요:\n\n{question}"
    else:
        prompt = question

    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    return response.content

def chat_interface():
    """ChatGPT처럼 대화할 수 있는 인터페이스"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 기존 채팅 기록 표시
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            with st.chat_message("user"):
                st.write(chat["message"])
        else:
            with st.chat_message("assistant"):
                st.write(chat["message"])

    # 대화 입력창
    if st.session_state.lang == 'korean':
        st.write("## ChatGPT와의 채팅 (GPT-4)")
        user_chat_input = st.chat_input("메시지를 입력하세요:")
    else:
        st.write("## Chat with ChatGPT (GPT-4)")
        user_chat_input = st.chat_input("Enter your message:")

    # 사용자가 입력하면 GPT 응답
    if user_chat_input:
        add_chat_message("user", user_chat_input)
        with st.chat_message("user"):
            st.write(user_chat_input)

        with st.spinner("GPT가 응답 중입니다..."):
            gpt_response = ask_gpt_question(user_chat_input, st.session_state.lang)
            add_chat_message("assistant", gpt_response)
            with st.chat_message("assistant"):
                st.write(gpt_response)

def pdf_to_text(upload_file):
    """PDF에서 텍스트 추출"""
    try:
        with pdfplumber.open(BytesIO(upload_file.getvalue())) as pdf:
            pages = []
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    pages.append(f"<PAGE{i+1}>\n{text}")
            return "\n".join(pages)
    except Exception as e:
        st.error(f"PDF에서 텍스트를 추출하는 중 오류가 발생했습니다: {e}")
        return ""

def pptx_to_text(upload_file):
    """PPTX에서 텍스트 추출"""
    try:
        prs = Presentation(BytesIO(upload_file.getvalue()))
        text_runs = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text_runs.append(shape.text)
        return "\n".join(text_runs)
    except Exception as e:
        st.error(f"PPTX에서 텍스트를 추출하는 중 오류가 발생했습니다: {e}")
        return ""

def image_to_text(uploaded_image):
    """이미지(캡처본 등)에서 텍스트(OCR) 추출"""
    try:
        image = Image.open(uploaded_image)
        # Tesseract 설치 및 경로 확인
        if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
            st.error("Tesseract가 설치되어 있지 않거나 경로가 올바르지 않습니다.")
            return ""
        # 'kor+eng'로 한국어+영어 혼합 인식 가능
        text = pytesseract.image_to_string(image, lang='kor+eng')
        return text
    except Exception as e:
        st.error(f'이미지에서 텍스트를 추출 중 오류가 발생했습니다: {e}')
        return ""

def hwp_to_text(upload_file):
    """HWP에서 텍스트 추출(hwp5txt 필요)"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.hwp') as tmp:
            tmp.write(upload_file.getvalue())
            tmp_path = tmp.name
        result = subprocess.run(["hwp5txt", tmp_path], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            st.error("HWP에서 텍스트를 추출할 수 없습니다. hwp5txt 툴이 설치되어 있는지 확인해주세요.")
            return ""
    except FileNotFoundError:
        st.error("hwp5txt 명령어를 찾을 수 없습니다. 설치 후 PATH 환경변수를 확인해주세요.")
        return ""
    except Exception as e:
        st.error(f"HWP 처리 중 오류가 발생했습니다: {e}")
        return ""

def doc_docx_to_text(upload_file, file_extension):
    """
    MS Word (doc, docx) 파일에서 텍스트 추출.
    - .docx: python-docx 이용
    - .doc: unoconv가 설치되어 있다면 .docx로 변환 후 추출
    """
    if file_extension == ".docx":
        # 바로 docx 처리
        try:
            doc = docx.Document(BytesIO(upload_file.getvalue()))
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            return "\n".join(full_text)
        except Exception as e:
            st.error(f"DOCX 처리 중 오류가 발생했습니다: {e}")
            return ""

    else:
        # .doc 파일 처리
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.doc') as tmp:
                tmp.write(upload_file.getvalue())
                tmp_path = tmp.name

            # unoconv를 이용해 doc -> docx 변환
            # 예: unoconv -f docx myfile.doc
            converted_path = tmp_path + ".docx"
            command = ["unoconv", "-f", "docx", tmp_path]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                st.error("unoconv를 이용해 .doc -> .docx 변환에 실패했습니다.")
                return ""

            # 변환된 docx 불러오기
            with open(converted_path, "rb") as f:
                converted_bytes = f.read()

            # docx 처리
            try:
                doc = docx.Document(BytesIO(converted_bytes))
                full_text = []
                for para in doc.paragraphs:
                    full_text.append(para.text)
                return "\n".join(full_text)
            except Exception as e:
                st.error(f"DOC -> DOCX 변환 후 처리 중 오류가 발생했습니다: {e}")
                return ""
            finally:
                # 임시 파일 정리
                try:
                    os.remove(converted_path)
                except:
                    pass

        except FileNotFoundError:
            st.error("unoconv 또는 LibreOffice가 설치되어 있지 않아 .doc 파일을 처리할 수 없습니다.")
            return ""
        except Exception as e:
            st.error(f"DOC 처리 중 오류가 발생했습니다: {e}")
            return ""

def detect_language(text):
    """GPT-4로 언어 감지"""
    llm = ChatOpenAI(
        model_name="gpt-4", 
        temperature=0, 
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    prompt = f"""다음 텍스트의 언어를 ISO 639-1 코드로 감지해 주세요 
(예: 'en'=영어, 'ko'=한국어 등):\n\n{text[:500]}"""
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    language_code = response.content.strip().lower().split()[0]
    return language_code

def summarize_text(text, language):
    """GPT-4로 요약"""
    llm = ChatOpenAI(
        model_name="gpt-4", 
        temperature=0,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    if language == 'korean':
        prompt = f"다음 텍스트를 읽고 서론, 본론, 결론으로 구성된 자세한 요약을 작성해 주세요:\n\n{text}"
    else:
        prompt = f"Read the following text and provide a detailed summary with introduction, main body, and conclusion:\n\n{text}"
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    return response.content.strip()

def extract_key_summary_words_with_sources(text, language):
    """중요 키워드(출처 포함) 추출"""
    llm = ChatOpenAI(
        model_name="gpt-4", 
        temperature=0,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    if language == 'korean':
        prompt = f"""다음 텍스트에서 중요한 키워드 5~10개를 추출하고, 
각 키워드의 출처(페이지 혹은 문맥)를 표시해주세요.

키워드1 (출처)
키워드2 (출처)
...

텍스트:
{text}"""
    else:
        prompt = f"""Extract 5 to 10 important keywords from the text and indicate their sources (page or context):

Keyword1 (Source)
Keyword2 (Source)
...

Text:
{text}"""
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    return response.content.strip()

def extract_and_search_terms(summary_text, extracted_text, language='english'):
    """요약된 내용에서 중요 용어와 위치 정보를 추출"""
    llm = ChatOpenAI(
        model_name="gpt-4", 
        temperature=0,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    if language == 'korean':
        prompt = f"다음 요약에서 중요한 용어 5~10개를 추출하고, 각 용어의 정의와 텍스트 내 페이지(혹은 위치) 정보를 제공:\n\n{summary_text}"
    else:
        prompt = f"From the following summary, extract 5-10 important terms, provide detailed definitions and their references in the text:\n\n{summary_text}"
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    return response.content.strip()

def generate_questions_for_user(text, language):
    """사용자가 깊이 생각할 수 있는 질문 3개 생성"""
    llm = ChatOpenAI(
        model_name="gpt-4", 
        temperature=0,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    if language == 'korean':
        prompt = f"다음 내용을 기반으로 사용자가 깊이 생각할 수 있는 질문 3개를 제시해 주세요:\n\n{text}"
    else:
        prompt = f"Based on the following content, generate 3 thoughtful questions for deeper understanding:\n\n{text}"
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    questions = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
    return questions

def create_ppt_from_text(text, filename="summary_output.pptx"):
    """요약 내용을 PPT로 변환"""
    prs = Presentation()
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    slide.shapes.title.text = "Summary"
    slide.placeholders[1].text = text

    buf = BytesIO()
    prs.save(buf)
    buf.seek(0)
    return buf

############################
# 메인 로직
############################

# 업로드 파일 처리 여부 상태
if "processed" not in st.session_state:
    st.session_state.processed = False

# 파일 업로더 (MS Word도 추가: doc, docx)
uploaded_file = st.file_uploader(
    "파일을 올려주세요 (PDF, PPTX, PNG, JPG, JPEG, HWP, DOC, DOCX 지원)",
    type=['pdf', 'pptx', 'png', 'jpg', 'jpeg', 'hwp', 'doc', 'docx']
)

# ChatGPT 대화 인터페이스
chat_interface()

if uploaded_file is not None:
    filename = uploaded_file.name
    extension = os.path.splitext(filename)[1].lower()

    # 업로드된 파일 해시 처리(중복 업로드 방지)
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    # 새로운 파일 업로드 시 세션 리셋
    if ("uploaded_file_hash" not in st.session_state or
        st.session_state.uploaded_file_hash != file_hash):
        st.session_state.uploaded_file_hash = file_hash
        st.session_state.extracted_text = ""
        st.session_state.summary = ""
        st.session_state.keywords = ""
        st.session_state.term_info = ""
        st.session_state.gpt_questions = []
        st.session_state.processed = False

    # 텍스트 추출 및 GPT 분석
    if not st.session_state.processed:
        if extension == ".pdf":
            extracted_text = pdf_to_text(uploaded_file)
        elif extension == ".pptx":
            extracted_text = pptx_to_text(uploaded_file)
        elif extension in [".png", ".jpg", ".jpeg"]:
            extracted_text = image_to_text(uploaded_file)  # OCR 기능
        elif extension == ".hwp":
            extracted_text = hwp_to_text(uploaded_file)
        elif extension in [".doc", ".docx"]:
            extracted_text = doc_docx_to_text(uploaded_file, extension)
        else:
            st.error("지원하지 않는 파일 형식입니다.")
            extracted_text = ""

        if not extracted_text.strip():
            st.error("업로드된 파일에서 텍스트를 추출할 수 없습니다.")
            st.session_state.summary = ""
        else:
            st.success("텍스트 추출 완료!")
            st.session_state.extracted_text = extracted_text

            # 언어 감지
            language_code = detect_language(extracted_text)
            if language_code == 'ko':
                lang = 'korean'
                language_name = '한국어'
            elif language_code == 'en':
                lang = 'english'
                language_name = '영어'
            else:
                lang = 'english'
                language_name = '알 수 없음 (영어로 진행)'

            st.write(f"### 감지된 언어: {language_name}")
            st.session_state.lang = lang

            # 요약 생성
            with st.spinner("요약 생성 중..."):
                summary = summarize_text(extracted_text, lang)
                st.session_state.summary = summary

            # 키워드(출처 포함) 추출
            with st.spinner("핵심 단어(출처 포함) 추출 중..."):
                key_summary_words = extract_key_summary_words_with_sources(extracted_text, lang)
                st.session_state.keywords = key_summary_words

            # 중요 단어 정보 추출
            with st.spinner("중요 단어 정보 추출 중..."):
                term_info = extract_and_search_terms(summary, extracted_text, language=lang)
                st.session_state.term_info = term_info

            # GPT가 사용자에게 질문 3개 생성
            with st.spinner("GPT가 사용자에게 물어볼 질문을 생성 중..."):
                gpt_questions = generate_questions_for_user(extracted_text, lang)
                st.session_state.gpt_questions = gpt_questions

            st.session_state.processed = True

    # 분석 결과 표시
    if st.session_state.get("processed", False):
        if 'summary' in st.session_state and st.session_state.summary.strip():
            st.write("## 요약 결과")
            st.write(st.session_state.summary)
        else:
            st.write("## 요약 결과를 표시할 수 없습니다.")

        if 'keywords' in st.session_state and st.session_state.keywords.strip():
            st.write("## 핵심 요약 단어 및 출처")
            st.write(st.session_state.keywords)

        if 'term_info' in st.session_state and st.session_state.term_info.strip():
            st.write("## 요약 내 중요한 단어 정보")
            st.write(st.session_state.term_info)

        st.write("---")

        # 요약 내용을 PPT로 다운로드
        if st.button("요약 내용을 PPT로 다운로드"):
            ppt_buffer = create_ppt_from_text(st.session_state.summary)
            st.download_button(
                label="PPT 다운로드",
                data=ppt_buffer,
                file_name="summary_output.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
            )

# 추출된 텍스트 검색 기능
if st.session_state.get("processed", False):
    st.write("---")
    if st.session_state.lang == 'korean':
        st.write("## 🔍 키워드 검색")
        search_query = st.text_input("검색할 키워드를 입력하세요:")
    else:
        st.write("## 🔍 Keyword Search")
        search_query = st.text_input("Enter a keyword to search:")

    if search_query:
        search_results = []
        for line in st.session_state.extracted_text.split('\n'):
            if search_query.lower() in line.lower():
                search_results.append(line.strip())

        if search_results:
            if st.session_state.lang == 'korean':
                st.write("### 검색 결과:")
            else:
                st.write("### Search Results:")
            for result in search_results:
                st.write(f"- {result}")
        else:
            if st.session_state.lang == 'korean':
                st.write("검색 결과가 없습니다.")
            else:
                st.write("No results found.")

# GPT가 사용자에게 질문 & 사용자 답변 후 피드백
if st.session_state.get("processed", False):
    st.write("---")
    if st.session_state.lang == 'korean':
        st.write("## GPT가 당신에게 질문합니다")
    else:
        st.write("## GPT has questions for you")

    if "gpt_questions" in st.session_state:
        for idx, question in enumerate(st.session_state.gpt_questions):
            user_answer = st.text_input(f"**{question}**", key=f"gpt_question_{idx}")
            if user_answer:
                with st.spinner("GPT가 응답 검토 중..."):
                    if st.session_state.lang == 'korean':
                        feedback_prompt = (
                            f"{question}\n\n"
                            f"사용자 답변: {user_answer}\n\n"
                            "피드백을 제공해 주세요."
                        )
                    else:
                        feedback_prompt = (
                            f"{question}\n\n"
                            f"User's answer: {user_answer}\n\n"
                            "Please provide feedback."
                        )
                    feedback = ask_gpt_question(feedback_prompt, st.session_state.lang)
                    if st.session_state.lang == 'korean':
                        st.write("### GPT의 피드백")
                    else:
                        st.write("### GPT's Feedback")
                    st.write(feedback)

st.write("---")
st.info("**주의:** ChatGPT는 때때로 부정확한 정보를 제공할 수 있으니, 중요한 내용은 별도로 검증하세요.")
