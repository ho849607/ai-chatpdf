import os
import shutil
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
import subprocess
import tempfile
import base64
import io

# docx2txt 설치 확인
try:
    import docx2txt
    DOCX_ENABLED = True
except ImportError:
    DOCX_ENABLED = False

# PaddleOCR 설치 확인
try:
    from paddleocr import PaddleOCR
    PADDLE_OCR_ENABLED = True
    # 한국어 OCR
    ocr = PaddleOCR(lang='ko')  
except ImportError:
    PADDLE_OCR_ENABLED = False

# NLTK 리소스 다운로드
nltk.download('punkt')
nltk.download('stopwords')

korean_stopwords = [
    '이', '그', '저', '것', '수', '등', '들', '및', '더', '로', '를', '에',
    '의', '은', '는', '가', '와', '과', '하다', '있다', '되다', '이다',
    '으로', '에서', '까지', '부터', '까지', '만', '하다', '그리고',
    '하지만', '그러나'
]

dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    openai_api_key = st.sidebar.text_input("OpenAI API 키를 입력하세요.", type="password")
    if not openai_api_key:
        st.error("OpenAI API 키가 설정되지 않았습니다.")
        st.stop()

openai.api_key = openai_api_key

# ---------------------------------------------------------------------
# 메인 타이틀
# ---------------------------------------------------------------------
st.title("studyhelper + Ctrl+V Image Chat Demo")
st.write("---")

if 'lang' not in st.session_state:
    st.session_state.lang = 'english'

st.warning("저작물을 불법 복제하여 게시하는 경우 당사는 책임지지 않으며, 저작권법에 유의하여 파일을 올려주세요.")

# 세션 초기화
if "processed" not in st.session_state:
    st.session_state.processed = False

# ------------------ 채팅 관련 세션/함수 ------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# **pasted_image_b64**: JS에서 postMessage로 넘겨줄 Base64 이미지를 임시 저장
if "pasted_image_b64" not in st.session_state:
    st.session_state.pasted_image_b64 = None

def add_chat_message(role: str, content):
    """채팅 기록에 메시지를 추가. content는 str(텍스트) 또는 PIL.Image"""
    st.session_state.chat_history.append({"role": role, "content": content})

def show_chat_history():
    """채팅 기록 표시"""
    for msg in st.session_state.chat_history:
        if msg["role"] == "assistant":
            with st.chat_message("assistant"):
                if isinstance(msg["content"], str):
                    st.write(msg["content"])
                elif isinstance(msg["content"], Image.Image):
                    st.image(msg["content"])
        else:  # user
            with st.chat_message("user"):
                if isinstance(msg["content"], str):
                    st.write(msg["content"])
                elif isinstance(msg["content"], Image.Image):
                    st.image(msg["content"])

# ------------------ OCR & 텍스트 추출 함수들 ------------------
def pdf_to_text(upload_file):
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
    """이미지 파일에서 텍스트 추출 (PaddleOCR 사용)"""
    if not PADDLE_OCR_ENABLED:
        st.warning("PaddleOCR가 설치되어 있지 않은 환경입니다. 이미지 텍스트 인식을 건너뜁니다.")
        return ""
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img:
            image = Image.open(uploaded_image)
            image.save(tmp_img.name, format='PNG')
            tmp_path = tmp_img.name
        # OCR 수행
        result = ocr.ocr(tmp_path, cls=False)
        extracted_text = ""
        for line in result:
            for word_info in line:
                extracted_text += word_info[1][0] + "\n"
        os.remove(tmp_path)
        return extracted_text.strip()
    except Exception as e:
        st.warning(f"이미지에서 텍스트를 추출할 수 없습니다: {e}")
        return ""

def hwp_to_text(upload_file):
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
        st.error("hwp5txt 명령어를 찾을 수 없습니다. hwp5txt가 제대로 설치되어 PATH에 포함되어 있는지 확인해주세요.")
        return ""
    except Exception as e:
        st.error(f"HWP 처리 중 오류가 발생했습니다: {e}")
        return ""

def docx_to_text(upload_file):
    if not DOCX_ENABLED:
        st.warning("docx2txt가 설치되어 있지 않아 .docx 파일을 처리할 수 없습니다.")
        return ""
    try:
        text = docx2txt.process(BytesIO(upload_file.getvalue()))
        return text if text else ""
    except Exception as e:
        st.error(f"DOCX 파일 처리 중 오류가 발생했습니다: {e}")
        return ""

def doc_to_text(upload_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.doc') as tmp:
            tmp.write(upload_file.getvalue())
            tmp_path = tmp.name
        result = subprocess.run(["antiword", tmp_path], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            st.error("DOC에서 텍스트를 추출할 수 없습니다. antiword 툴이 설치되어 있는지 확인해주세요.")
            return ""
    except FileNotFoundError:
        st.error("antiword 명령어를 찾을 수 없습니다. antiword가 제대로 설치되어 PATH에 포함되어 있는지 확인해주세요.")
        return ""
    except Exception as e:
        st.error(f"DOC 처리 중 오류가 발생했습니다: {e}")
        return ""

# ------------------ GPT 호출 함수들 ------------------
def ask_gpt_model(messages):
    llm = ChatOpenAI(
        model_name="gpt-4", 
        temperature=0,
        streaming=True, 
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    response = llm(messages)
    return response.content

def detect_language(text):
    prompt = f"다음 텍스트의 언어를 ISO 639-1 코드로 감지해 주세요 (예: 'en'은 영어, 'ko'는 한국어):\n\n{text[:500]}"
    messages = [HumanMessage(content=prompt)]
    return ask_gpt_model(messages).strip().lower().split()[0]

def summarize_text(text, language):
    if language == 'korean':
        prompt = f"다음 텍스트를 읽고 서론, 본론, 결론으로 구성된 자세한 요약을 작성해 주세요:\n\n{text}"
    else:
        prompt = f"Read the following text and write a detailed summary with introduction, main body, and conclusion:\n\n{text}"
    messages = [HumanMessage(content=prompt)]
    return ask_gpt_model(messages).strip()

def extract_key_summary_words_with_sources(text, language):
    if language == 'korean':
        prompt = f"""다음 텍스트에서 중요한 키워드 5~10개를 추출하고, 각 키워드의 출처를 표시해주세요.

키워드1 (출처)
키워드2 (출처)
...

텍스트:
{text}"""
    else:
        prompt = f"""Extract 5 to 10 important keywords from the text and indicate their sources:

Keyword1 (Source)
Keyword2 (Source)
...

Text:
{text}"""
    messages = [HumanMessage(content=prompt)]
    return ask_gpt_model(messages).strip()

def extract_and_search_terms(summary_text, extracted_text, language='english'):
    if language == 'korean':
        prompt = f"다음 요약에서 중요한 용어 5~10개를 추출하고, 각 용어 정의와 텍스트 내 페이지 정보를 제공:\n\n{summary_text}"
    else:
        prompt = f"From the following summary, extract 5-10 important terms, provide detailed definitions and their page references:\n\n{summary_text}"
    messages = [HumanMessage(content=prompt)]
    return ask_gpt_model(messages).strip()

def generate_questions_for_user(text, language):
    if language == 'korean':
        prompt = f"다음 내용을 기반으로 사용자가 깊이 생각할 수 있는 질문 3개 제시:\n\n{text}"
    else:
        prompt = f"Based on the following content, generate 3 thoughtful questions for deeper understanding:\n\n{text}"
    messages = [HumanMessage(content=prompt)]
    raw_response = ask_gpt_model(messages)
    return [q.strip() for q in raw_response.strip().split('\n') if q.strip()]

def create_ppt_from_text(text, filename="summary_output.pptx"):
    prs = Presentation()
    title_slide_layout = prs.slides.add_slide(title_slide_layout)
    slide = prs.slides.add_slide(title_slide_layout)
    slide.shapes.title.text = "Summary"
    slide.placeholders[1].text = text

    buf = BytesIO()
    prs.save(buf)
    buf.seek(0)
    return buf

# ---------------------------------------------------------------------
# 1) 채팅창 UI (텍스트 + Ctrl+V)
# ---------------------------------------------------------------------
# 채팅 기록 표시
show_chat_history()

# 자바스크립트 삽입 (Ctrl+V -> paste event)
paste_js_code = """
<script>
document.addEventListener('paste', async (event) => {
    const items = (event.clipboardData || event.originalEvent.clipboardData).items;
    if (!items) return;
    for (let idx in items) {
        let item = items[idx];
        if (item.kind === 'file') {
            let blob = item.getAsFile();
            if (blob) {
                // blob -> base64
                const reader = new FileReader();
                reader.onload = function(e) {
                    const base64Data = e.target.result.split(',')[1];
                    // postMessage로 streamlit에 전송
                    window.parent.postMessage(
                        { 
                          type: 'PASTE_IMAGE', 
                          base64: base64Data 
                        }, 
                        '*'
                    );
                };
                reader.readAsDataURL(blob);
            }
        }
    }
});
</script>
"""

# HTML 컴포넌트 삽입
st.components.v1.html(paste_js_code, height=0)

st.info("**Ctrl+V**로 이미지를 붙여넣은 뒤, 아래 ‘붙여넣기 이미지 처리’ 버튼을 누르세요. (데모)")

# (임시) "붙여넣기 이미지 처리" 버튼
def handle_pasted_image():
    """세션에 저장된 base64 -> PIL.Image로 변환하여 채팅에 추가"""
    if st.session_state.pasted_image_b64:
        image_data = base64.b64decode(st.session_state.pasted_image_b64)
        image = Image.open(io.BytesIO(image_data))
        add_chat_message("user", image)
        add_chat_message("assistant", "Ctrl+V로 이미지를 잘 받았습니다!")
        st.session_state.pasted_image_b64 = None

if st.button("붙여넣기 이미지 처리"):
    handle_pasted_image()

# 채팅 텍스트 입력
user_text = st.chat_input("메시지를 입력하세요...")
if user_text:
    add_chat_message("user", user_text)
    add_chat_message("assistant", f"사용자 입력: {user_text}")

# 채팅 기록 다시 표시 (업데이트된 내용 포함)
show_chat_history()

# ---------------------------------------------------------------------
# 2) 파일 업로드 -> 텍스트 추출 -> GPT 분석
# ---------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "파일 업로드 (PDF, PPTX, PNG, JPG, JPEG, HWP, DOC, DOCX)",
    type=['pdf', 'pptx', 'png', 'jpg', 'jpeg', 'hwp', 'doc', 'docx']
)

if uploaded_file is not None:
    filename = uploaded_file.name
    extension = os.path.splitext(filename)[1].lower()

    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    # 업로드 파일이 새로 바뀌었는지 확인
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
            extracted_text = image_to_text(uploaded_file)
        elif extension == ".hwp":
            extracted_text = hwp_to_text(uploaded_file)
        elif extension == ".docx":
            extracted_text = docx_to_text(uploaded_file)
        elif extension == ".doc":
            extracted_text = doc_to_text(uploaded_file)
        else:
            st.error("지원하지 않는 파일 형식입니다.")
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
                language_name = '알 수 없음 (영어 진행)'

            st.write(f"### 감지된 언어: {language_name}")
            st.session_state.lang = lang
            st.session_state.extracted_text = extracted_text

            with st.spinner("요약 생성 중..."):
                summary = summarize_text(extracted_text, lang)
                st.session_state.summary = summary

            with st.spinner("핵심 단어 추출 중..."):
                key_summary_words = extract_key_summary_words_with_sources(st.session_state.summary, lang)
                st.session_state.keywords = key_summary_words

            with st.spinner("중요 단어 정보 추출 중..."):
                term_info = extract_and_search_terms(st.session_state.summary, extracted_text, language=lang)
                st.session_state.term_info = term_info

            with st.spinner("GPT가 질문을 생성 중..."):
                gpt_questions = generate_questions_for_user(extracted_text, lang)
                st.session_state.gpt_questions = gpt_questions

            st.session_state.processed = True

    if st.session_state.get("processed", False):
        # 결과 표시
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

# ---------------------------------------------------------------------
# 3) 키워드 검색 & GPT가 사용자에게 질문
# ---------------------------------------------------------------------
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
                with st.spinner("GPT가 응답을 검토 중입니다..."):
                    if st.session_state.lang == 'korean':
                        feedback_prompt = f"{question}\n\n사용자 답변: {user_answer}\n\n피드백을 제공해 주세요."
                    else:
                        feedback_prompt = f"{question}\n\nUser's answer: {user_answer}\n\nPlease provide feedback on this."
                    feedback_answer = ask_gpt_model([HumanMessage(content=feedback_prompt)])
                    if st.session_state.lang == 'korean':
                        st.write("### GPT의 피드백")
                    else:
                        st.write("### GPT's Feedback")
                    st.write(feedback_answer)


# ---------------------------------------------------------------------
# 마지막 안내
# ---------------------------------------------------------------------
st.write("---")
st.info("""
**Ctrl+V를 통해 이미지를 붙여넣은 뒤, '붙여넣기 이미지 처리' 버튼을 눌러주세요.**  
이는 Streamlit에서 공식적으로 지원하지 않는 기능을 **자바스크립트 + postMessage**로 간단히 흉내 낸 예시입니다.  
실제 서비스에서는 **커스텀 컴포넌트**로 브라우저/파이썬 간 실시간 통신을 구성해야 합니다.
""")
