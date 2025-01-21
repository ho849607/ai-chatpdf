import os
import shutil
import streamlit as st
from io import BytesIO
from dotenv import load_dotenv
import pdfplumber
from pptx import Presentation
import openai
from pathlib import Path
import hashlib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from PIL import Image
import cv2
import numpy as np
import subprocess
import tempfile

# docx2txt 설치 확인 (설치 안 된 경우 except로 처리)
try:
    import docx2txt
    DOCX_ENABLED = True
except ImportError:
    DOCX_ENABLED = False

# PaddleOCR 설치 확인 (설치 안 된 경우 except로 처리)
try:
    from paddleocr import PaddleOCR
    PADDLE_OCR_ENABLED = True
    ocr = PaddleOCR(lang='ko')
except ImportError:
    PADDLE_OCR_ENABLED = False

# (1) NLTK_DATA 경로를 /tmp 로 지정 (쓰기 가능)
nltk_data_dir = "/tmp/nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)
os.environ["NLTK_DATA"] = nltk_data_dir
nltk.data.path.append(nltk_data_dir)

# 필요한 nltk 리소스 다운로드
nltk.download("punkt", download_dir=nltk_data_dir)
nltk.download("stopwords", download_dir=nltk_data_dir)

korean_stopwords = [
    '이', '그', '저', '것', '수', '등', '들', '및', '더', '로', '를', '에',
    '의', '은', '는', '가', '와', '과', '하다', '있다', '되다', '이다',
    '으로', '에서', '까지', '부터', '만', '그리고', '하지만', '그러나'
]
english_stopwords = set(stopwords.words('english'))
korean_stopwords_set = set(korean_stopwords)
final_stopwords = english_stopwords.union(korean_stopwords_set)

# .env 로드
dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    openai_api_key = st.sidebar.text_input("OpenAI API 키를 입력하세요.", type="password")
    if not openai_api_key:
        st.error("OpenAI API 키가 설정되지 않았습니다.")
        st.stop()

openai.api_key = openai_api_key

st.set_page_config(page_title="studyhelper")
st.title("studyhelper")
st.write("---")

if 'lang' not in st.session_state:
    st.session_state.lang = 'english'

st.warning("저작물을 불법 복제하여 게시하는 경우 당사는 책임지지 않으며, 저작권법에 유의해 주세요.")

################################################################################
# (A) 최신 openai.ChatCompletion 사용 함수
################################################################################
def call_openai_chat(messages, model="gpt-4", temperature=0.0):
    """
    openai>=1.0.0 용 법 (APIRemovedInV1 오류 없음)
    messages: [{"role": "user", "content": "..."} ...]
    """
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message.content.strip()

################################################################################
# (B) 이전 langchain.ChatOpenAI 대신 직접 GPT 호출하는 함수들
################################################################################
def ask_gpt_question(user_question, language='korean'):
    """
    사용자 질문 -> GPT (gpt-4) -> 답변
    """
    if language == 'korean':
        system_prompt = "당신은 유용한 AI 어시스턴트입니다."
    else:
        system_prompt = "You are a helpful AI assistant."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ]
    return call_openai_chat(messages, model="gpt-4", temperature=0.0)

def detect_language(text):
    """
    텍스트의 언어 감지 -> ISO 639-1 코드('ko' or 'en' 등)
    """
    system_prompt = "You are a language detection assistant."
    prompt = f"다음 텍스트의 언어를 ISO 639-1 코드로 감지해 주세요:\n\n{text[:500]}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    result = call_openai_chat(messages, model="gpt-4", temperature=0.0)
    lang_code = result.strip().lower().split()[0]
    return lang_code

def summarize_text(text, language):
    """
    텍스트 요약 -> 서론, 본론, 결론 구조
    """
    if language == 'korean':
        user_prompt = f"다음 텍스트를 읽고 서론, 본론, 결론으로 구성된 자세한 요약을 작성해 주세요:\n\n{text}"
    else:
        user_prompt = f"Read the following text and write a detailed summary with introduction, main body, and conclusion:\n\n{text}"

    messages = [
        {"role": "system", "content": "You are a summarization AI assistant."},
        {"role": "user", "content": user_prompt}
    ]
    return call_openai_chat(messages, model="gpt-4", temperature=0.0)

def extract_key_summary_words_with_sources(text, language):
    """
    텍스트에서 중요한 키워드 5~10 추출 + 출처 표시
    """
    if language == 'korean':
        user_prompt = f"""다음 텍스트에서 중요한 키워드 5~10개를 추출하고, 각 키워드의 출처를 표시해주세요.

키워드1 (출처)
키워드2 (출처)
...

텍스트:
{text}"""
    else:
        user_prompt = f"""Extract 5 to 10 important keywords from the text and indicate their sources:

Keyword1 (Source)
Keyword2 (Source)
...

Text:
{text}"""
    messages = [
        {"role": "system", "content": "You are a keyword extraction assistant."},
        {"role": "user", "content": user_prompt}
    ]
    return call_openai_chat(messages, model="gpt-4", temperature=0.0)

def extract_and_search_terms(summary_text, extracted_text, language='english'):
    """
    요약에서 중요한 용어 5~10개를 추출 + 정의 + 텍스트 내 페이지 정보
    """
    if language == 'korean':
        user_prompt = f"다음 요약에서 중요한 용어 5~10개를 추출하고, 각 용어 정의와 텍스트 내 페이지 정보를 제공:\n\n{summary_text}"
    else:
        user_prompt = f"From the following summary, extract 5-10 important terms, provide detailed definitions and their page references:\n\n{summary_text}"

    messages = [
        {"role": "system", "content": "You are an AI assistant for term extraction."},
        {"role": "user", "content": user_prompt}
    ]
    return call_openai_chat(messages, model="gpt-4", temperature=0.0)

def generate_questions_for_user(text, language):
    """
    사용자에게 3개 질문 (더 깊이 생각할 수 있도록)
    """
    if language == 'korean':
        user_prompt = f"다음 내용을 기반으로 사용자가 깊이 생각할 수 있는 질문 3개 제시:\n\n{text}"
    else:
        user_prompt = f"Based on the following content, generate 3 thoughtful questions:\n\n{text}"

    messages = [
        {"role": "system", "content": "You are an AI that creates thoughtful questions."},
        {"role": "user", "content": user_prompt}
    ]
    result = call_openai_chat(messages, model="gpt-4", temperature=0.0)
    questions = [q.strip() for q in result.split('\n') if q.strip()]
    return questions

################################################################################
# (C) 이미지 처리 (PaddleOCR + OpenCV)
################################################################################
def image_opencv_demo(uploaded_image):
    image = Image.open(uploaded_image).convert("RGB")
    img_array = np.array(image)

    # 색상 반전
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    inverted_bgr = 255 - img_bgr
    inverted_rgb = cv2.cvtColor(inverted_bgr, cv2.COLOR_BGR2RGB)

    # 화면 표시
    st.image(inverted_rgb, caption="색상 반전(OpenCV)", use_column_width=True)

    # PIL로 변환 후 반환
    return Image.fromarray(inverted_rgb)

def image_to_text(uploaded_image):
    """
    (반전이미지 + OCR 텍스트) 반환 예시
    """
    if not PADDLE_OCR_ENABLED:
        st.warning("PaddleOCR 미설치. 이미지 텍스트 인식을 건너뜁니다.")
        inverted = image_opencv_demo(uploaded_image)
        return inverted, ""

    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img:
            image = Image.open(uploaded_image)
            image.save(tmp_img.name, format='PNG')
            tmp_path = tmp_img.name

        result = ocr.ocr(tmp_path, cls=False)
        extracted_text = ""
        for line in result:
            for word_info in line:
                extracted_text += word_info[1][0] + "\n"

        os.remove(tmp_path)
        inverted_pil = image_opencv_demo(uploaded_image)
        return inverted_pil, extracted_text.strip()

    except Exception as e:
        st.warning(f"OCR 오류: {e}")
        inverted_pil = image_opencv_demo(uploaded_image)
        return inverted_pil, ""

################################################################################
# (D) PDF, PPTX, DOCX, HWP 등 텍스트 추출
################################################################################
def pdf_to_text(upload_file):
    try:
        with pdfplumber.open(BytesIO(upload_file.getvalue())) as pdf:
            pages = [page.extract_text() for page in pdf.pages]
        return "\n".join(p for p in pages if p)
    except Exception as e:
        st.error(f"PDF 텍스트 추출 오류: {e}")
        return ""

def pptx_to_text(upload_file):
    try:
        prs = Presentation(BytesIO(upload_file.getvalue()))
        text_runs = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text_runs.append(shape.text)
        return "\n".join(text_runs)
    except Exception as e:
        st.error(f"PPTX 텍스트 추출 오류: {e}")
        return ""

def hwp_to_text(upload_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.hwp') as tmp:
            tmp.write(upload_file.getvalue())
            tmp_path = tmp.name
        # hwp5txt 필요 (별도 설치)
        result = subprocess.run(["hwp5txt", tmp_path], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            st.error("HWP 텍스트 추출 실패. hwp5txt 설치 필요.")
            return ""
    except FileNotFoundError:
        st.error("hwp5txt 명령어를 찾을 수 없습니다.")
        return ""
    except Exception as e:
        st.error(f"HWP 처리 오류: {e}")
        return ""

################################################################################
# (E) Chat 인터페이스
################################################################################
def add_chat_message(role, message):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({"role": role, "message": message})

def chat_interface():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            with st.chat_message("user"):
                st.write(chat["message"])
        else:
            with st.chat_message("assistant"):
                st.write(chat["message"])

    # 한글 환경 가정
    user_chat_input = st.chat_input("질문을 입력하세요:")
    if user_chat_input:
        add_chat_message("user", user_chat_input)
        with st.chat_message("user"):
            st.write(user_chat_input)

        with st.spinner("GPT가 답변 중..."):
            gpt_response = ask_gpt_question(user_chat_input, language='korean')
            add_chat_message("assistant", gpt_response)
            with st.chat_message("assistant"):
                st.write(gpt_response)

################################################################################
# PPT 다운로드 함수
################################################################################
def create_ppt_from_text(text, filename="summary_output.pptx"):
    prs = Presentation()
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    slide.shapes.title.text = "Summary"
    slide.placeholders[1].text = text

    buf = BytesIO()
    prs.save(buf)
    buf.seek(0)
    return buf

################################################################################
# 메인 (파일 업로드 & 처리)
################################################################################
def main():
    uploaded_file = st.file_uploader(
        "파일을 업로드하세요 (PDF, PPTX, 이미지, HWP, DOC, DOCX)",
        type=['pdf', 'pptx', 'png', 'jpg', 'jpeg', 'hwp', 'doc', 'docx']
    )

    # GPT-4와의 채팅
    chat_interface()

    if uploaded_file is not None:
        filename = uploaded_file.name
        extension = os.path.splitext(filename)[1].lower()

        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()

        # 세션 중복 업로드 체크
        if ("uploaded_file_hash" not in st.session_state or
            st.session_state.uploaded_file_hash != file_hash):
            st.session_state.uploaded_file_hash = file_hash
            st.session_state.extracted_text = ""
            st.session_state.summary = ""
            st.session_state.keywords = ""
            st.session_state.term_info = ""
            st.session_state.gpt_questions = []
            st.session_state.processed = False

        if not st.session_state.processed:
            if extension == ".pdf":
                extracted_text = pdf_to_text(uploaded_file)
            elif extension == ".pptx":
                extracted_text = pptx_to_text(uploaded_file)
            elif extension in [".png", ".jpg", ".jpeg"]:
                # 이미지 → (반전, OCR)
                inverted_img, ocr_text = image_to_text(uploaded_file)
                extracted_text = ocr_text
            elif extension == ".hwp":
                extracted_text = hwp_to_text(uploaded_file)
            elif extension == ".docx":
                extracted_text = docx_to_text(uploaded_file)
            elif extension == ".doc":
                # antiword
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.doc') as tmp:
                        tmp.write(file_bytes)
                        tmp_path = tmp.name
                    result = subprocess.run(["antiword", tmp_path], capture_output=True, text=True)
                    if result.returncode == 0:
                        extracted_text = result.stdout
                    else:
                        st.error("DOC 텍스트 추출 실패. antiword 설치 필요.")
                        extracted_text = ""
                except FileNotFoundError:
                    st.error("antiword 명령어가 없습니다.")
                    extracted_text = ""
                except Exception as e:
                    st.error(f"DOC 처리 오류: {e}")
                    extracted_text = ""
            else:
                st.error("지원하지 않는 형식입니다.")
                extracted_text = ""

            if not extracted_text.strip():
                st.error("업로드된 파일에서 텍스트를 추출할 수 없습니다.")
                st.session_state.summary = ""
            else:
                st.success("텍스트 추출 완료!")
                language_code = detect_language(extracted_text)
                if language_code == 'ko':
                    lang = 'korean'
                    language_name = '한국어'
                elif language_code == 'en':
                    lang = 'english'
                    language_name = '영어'
                else:
                    lang = 'english'
                    language_name = f'알 수 없음(기본 영어)'

                st.write(f"감지된 언어: {language_name}")
                st.session_state.lang = lang
                st.session_state.extracted_text = extracted_text

                # 요약
                with st.spinner("요약 생성 중..."):
                    summary = summarize_text(extracted_text, lang)
                    st.session_state.summary = summary

                # 핵심 단어
                with st.spinner("핵심 단어 추출 중..."):
                    key_summary_words = extract_key_summary_words_with_sources(st.session_state.summary, lang)
                    st.session_state.keywords = key_summary_words

                # 중요 단어 정보
                with st.spinner("중요 단어 정보 추출 중..."):
                    term_info = extract_and_search_terms(st.session_state.summary, extracted_text, language=lang)
                    st.session_state.term_info = term_info

                # GPT가 사용자에게 질문
                with st.spinner("GPT가 질문 생성 중..."):
                    gpt_questions = generate_questions_for_user(extracted_text, lang)
                    st.session_state.gpt_questions = gpt_questions

            st.session_state.processed = True

        # 결과 표시
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
            if st.button("요약 내용을 PPT로 다운로드"):
                ppt_buffer = create_ppt_from_text(st.session_state.summary)
                st.download_button(
                    label="PPT 다운로드",
                    data=ppt_buffer,
                    file_name="summary_output.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                )

if __name__ == "__main__":
    main()
