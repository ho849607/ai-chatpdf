import os
import nltk

# (1) NLTK_DATA 경로를 /tmp 로 지정 (쓰기 가능)
nltk_data_dir = "/tmp/nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)  # 디렉토리가 없으면 생성

# NLTK가 /tmp/nltk_data를 참조하도록 설정
os.environ["NLTK_DATA"] = nltk_data_dir
nltk.data.path.append(nltk_data_dir)

# stopwords 다운로드 시도
nltk.download("stopwords", download_dir=nltk_data_dir)

import streamlit as st
from io import BytesIO
from dotenv import load_dotenv
import openai
from pathlib import Path
import hashlib

# 뒤쪽에 nltk 재-import가 있어도 충돌은 없으니 그대로 둬도 됩니다.
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# docx2txt 설치 확인
try:
    import docx2txt
    DOCX_ENABLED = True
except ImportError:
    DOCX_ENABLED = False

# PyPDF2 설치 확인
try:
    import PyPDF2
    PDF_ENABLED = True
except ImportError:
    PDF_ENABLED = False

# 초기 NLTK 다운로드 (tokenizer, stopwords가 없는 경우 다운로드)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_dir)

# 사용자 정의 한국어 스톱워드
korean_stopwords = [
    '이', '그', '저', '것', '수', '등', '들', '및', '더', '로', '를', '에',
    '의', '은', '는', '가', '와', '과', '하다', '있다', '되다', '이다',
    '으로', '에서', '까지', '부터', '만', '그리고', '하지만', '그러나'
]
# NLTK 영어 스톱워드 + 한국어 스톱워드 병합
english_stopwords = set(stopwords.words('english'))
korean_stopwords_set = set(korean_stopwords)
final_stopwords = english_stopwords.union(korean_stopwords_set)

st.set_page_config(page_title="studyhelper", layout="centered")

###############################################################################
# .env 로드 및 OpenAI API 키 설정
###############################################################################
dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    openai_api_key = st.sidebar.text_input("OpenAI API 키를 입력하세요.", type="password")
    if not openai_api_key:
        st.error("OpenAI API 키가 설정되지 않았습니다.")
        st.stop()

# OpenAI API 키 직접 설정
openai.api_key = openai_api_key

###############################################################################
# GPT 연동 함수 (구버전 ChatCompletion)
###############################################################################
def ask_gpt(prompt_text, model_name="gpt-4", temperature=0.0):
    """
    openai==0.28.x 이하 버전에서:
    구버전 API -> openai.ChatCompletion.create(...)
    """
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt_text}
        ],
        temperature=temperature
    )
    return response.choices[0].message["content"].strip()

###############################################################################
# 채팅 인터페이스
###############################################################################
def add_chat_message(role, message):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({"role": role, "message": message})

def chat_interface():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 기존 채팅 이력 표시
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            with st.chat_message("user"):
                st.write(chat["message"])
        else:
            with st.chat_message("assistant"):
                st.write(chat["message"])

    user_chat_input = st.chat_input("메시지를 입력하세요:")
    if user_chat_input:
        add_chat_message("user", user_chat_input)
        with st.chat_message("user"):
            st.write(user_chat_input)

        with st.spinner("GPT가 응답 중입니다..."):
            gpt_response = ask_gpt(user_chat_input, model_name="gpt-4", temperature=0.0)
            add_chat_message("assistant", gpt_response)
            with st.chat_message("assistant"):
                st.write(gpt_response)

###############################################################################
# DOCX 텍스트 추출 함수 - docx 파일 전용
###############################################################################
def docx_to_text(upload_file):
    if not DOCX_ENABLED:
        st.warning("docx2txt가 설치되어 있지 않아 .docx 파일을 처리할 수 없습니다.")
        return ""
    try:
        import docx2txt
        text = docx2txt.process(BytesIO(upload_file.getvalue()))
        return text if text else ""
    except Exception as e:
        st.error(f"DOCX 파일 처리 중 오류가 발생했습니다: {e}")
        return ""

###############################################################################
# 파일 형식에 따른 텍스트 추출 함수 (docx와 pdf 지원)
###############################################################################
def extract_text_from_file(upload_file):
    filename = upload_file.name
    ext = filename.split('.')[-1].lower()

    if ext == "docx":
        # DOCX 처리
        return docx_to_text(upload_file)

    elif ext == "pdf":
        # PDF 처리
        if not PDF_ENABLED:
            st.error("PyPDF2가 설치되어 있지 않아 PDF 파일을 처리할 수 없습니다. "
                     "설치 후 다시 시도해주세요. (예: pip install PyPDF2)")
            return ""
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(BytesIO(upload_file.getvalue()))
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text.strip()
        except Exception as e:
            st.error(f"PDF 파일 처리 중 오류가 발생했습니다: {e}")
            return ""
    else:
        st.error("지원하지 않는 파일 형식입니다. 지원되는 형식: docx, pdf.")
        return ""

###############################################################################
# GPT를 이용한 문서 분석 함수 (요약, 중요 내용 추출, 질문 생성)
###############################################################################
def analyze_document_text(doc_text):
    # 문서 요약
    prompt_summary = f"""
    아래 문서를 읽고, 핵심 내용을 간략하게 요약해 주세요.
    
    문서:
    {doc_text}
    """
    summary = ask_gpt(prompt_summary, "gpt-4", 0.3)

    # 중요한 내용 추출
    prompt_important = f"""
    아래 문서에서 중요한 정보, 핵심 아이디어, 그리고 주목할 만한 내용을 추출해 주세요.
    
    문서:
    {doc_text}
    """
    important_content = ask_gpt(prompt_important, "gpt-4", 0.3)

    # 질문 생성 (사용자가 스스로 생각해볼 수 있는 질문)
    prompt_questions = f"""
    위 문서를 기반으로 독자가 스스로 답변해 볼 수 있는 질문 3~5개를 생성해 주세요.
    질문들은 문서의 핵심 내용과 관련되어야 합니다.
    
    문서:
    {doc_text}
    """
    questions = ask_gpt(prompt_questions, "gpt-4", 0.3)

    return {
        "summary": summary,
        "important_content": important_content,
        "questions": questions
    }

###############################################################################
# (참고용) 기존 DOCX 분석 함수 - 간단 예시
###############################################################################
def analyze_docx_text(docx_text):
    """
    docx 텍스트가 업로드되면 자동으로 핵심 내용, 특징, 개선점 등을 요약하여
    보여주는 간단한 예시 GPT 분석 함수.
    """
    prompt = f"""
    아래는 사용자가 업로드한 DOCX 원본 텍스트입니다.
    주요 핵심 내용, 중요한 아이디어나 요점이 있다면 알려주세요.
    간단한 요약과 함께 추가적인 분석 및 개선사항도 제안해 주세요.

    원문:
    {docx_text}
    """
    analysis = ask_gpt(prompt, "gpt-4", 0.3)
    return analysis

###############################################################################
# 커뮤니티(아이디어 공유 & 투자)
###############################################################################
def community_investment_tab():
    st.header("아이디어 공유 & 투자 커뮤니티")

    if "community_ideas" not in st.session_state:
        st.session_state.community_ideas = []

    st.subheader("새로운 아이디어 제안하기")
    idea_title = st.text_input("아이디어 제목", "")
    idea_content = st.text_area("아이디어 내용(간략 소개)", "")

    if st.button("아이디어 등록"):
        if idea_title.strip() and idea_content.strip():
            new_idea = {
                "title": idea_title,
                "content": idea_content,
                "comments": [],
                "likes": 0,
                "dislikes": 0,
                "investment": 0
            }

            # AI 분석/개선 요약 (자동)
            with st.spinner("아이디어 분석/개선 중..."):
                auto_analysis_prompt = f"""
                다음 아이디어를 짧게 분석하고, 핵심 내용을 요약한 뒤 
                개선점을 제안해 주세요.

                아이디어:
                {idea_content}
                """
                analysis_result = ask_gpt(auto_analysis_prompt, "gpt-4", 0.3)
                new_idea["auto_analysis"] = analysis_result

            st.session_state.community_ideas.append(new_idea)
            st.success("아이디어가 등록되었습니다! (자동 분석/개선 결과 포함)")
        else:
            st.warning("제목과 내용을 입력하세요.")

    st.write("---")
    st.subheader("커뮤니티 아이디어 목록")

    if len(st.session_state.community_ideas) == 0:
        st.write("아직 등록된 아이디어가 없습니다.")
    else:
        for idx, idea in enumerate(st.session_state.community_ideas):
            with st.expander(f"{idx+1}. {idea['title']}"):
                st.write(f"**내용**: {idea['content']}")
                # 자동 분석/개선 결과
                if "auto_analysis" in idea and idea["auto_analysis"].strip():
                    st.write("**AI 자동 분석/개선 요약**:")
                    st.write(idea["auto_analysis"])

                # 3개 컬럼(좋아요/싫어요/투자)
                col1, col2, col3, col4 = st.columns([1,1,2,1])
                with col1:
                    st.write(f"👍 좋아요: {idea['likes']}")
                    if st.button(f"좋아요 (아이디어 #{idx+1})"):
                        idea["likes"] += 1
                        st.experimental_rerun()

                with col2:
                    st.write(f"👎 싫어요: {idea['dislikes']}")
                    if st.button(f"싫어요 (아이디어 #{idx+1})"):
                        idea["dislikes"] += 1
                        st.experimental_rerun()

                with col3:
                    st.write(f"💰 현재 투자액: {idea['investment']}")
                    invest_amount = st.number_input(
                        f"투자 금액 (아이디어 #{idx+1})",
                        min_value=0,
                        step=10,
                        key=f"investment_input_{idx}"
                    )
                    if st.button(f"투자하기 (아이디어 #{idx+1})"):
                        idea["investment"] += invest_amount
                        st.success(f"{invest_amount}만큼 투자했습니다!")
                        st.experimental_rerun()

                # 휴지통 버튼으로 삭제
                with col4:
                    if st.button(f"🗑 (아이디어 #{idx+1})"):
                        st.session_state.community_ideas.pop(idx)
                        st.experimental_rerun()

                # 댓글
                st.write("### 댓글")
                if len(idea["comments"]) == 0:
                    st.write("아직 댓글이 없습니다.")
                else:
                    for c_idx, comment in enumerate(idea["comments"]):
                        st.write(f"- {comment}")

                comment_text = st.text_input(
                    f"댓글 달기 (아이디어 #{idx+1})",
                    key=f"comment_input_{idx}"
                )
                if st.button(f"댓글 등록 (아이디어 #{idx+1})"):
                    if comment_text.strip():
                        idea["comments"].append(comment_text.strip())
                        st.success("댓글이 등록되었습니다!")
                        st.experimental_rerun()
                    else:
                        st.warning("댓글 내용을 입력하세요.")

                st.write("---")
                st.write("### (추가) GPT 버튼 기능들")

                if st.button(f"SWOT 분석 (아이디어 #{idx+1})"):
                    with st.spinner("SWOT 분석 중..."):
                        prompt_swot = f"""
                        아래 아이디어에 대해 간략하게 SWOT(Strengths, Weaknesses, Opportunities, Threats)을 해주세요.

                        아이디어:
                        {idea['content']}
                        """
                        swot_result = ask_gpt(prompt_swot, "gpt-4", 0.3)
                        st.write("**SWOT 분석 결과**:")
                        st.write(swot_result)

                if st.button(f"주제별 분류 (아이디어 #{idx+1})"):
                    with st.spinner("아이디어 주제 분류 중..."):
                        prompt_category = f"""
                        아래 아이디어가 어느 분야(기술, 푸드, 교육, 금융, 건강, 기타)인지 추정해 주세요.
                        간단한 근거와 함께 알려주면 감사하겠습니다.

                        아이디어:
                        {idea['content']}
                        """
                        category_result = ask_gpt(prompt_category, "gpt-4", 0.3)
                        st.write("**주제별 분류 결과**:")
                        st.write(category_result)

                if st.button(f"AI 아이디어 추가 개선 (아이디어 #{idx+1})"):
                    with st.spinner("AI가 아이디어 추가 개선/분석 중..."):
                        prompt_improve = f"""
                        아래 아이디어가 있습니다. 이 아이디어를 좀 더 구체적이고 발전된 방향으로 개선하거나 
                        보완할 점, 참고해야 할 사항, 필요한 기술이나 리소스 등을 제안해 주세요.

                        아이디어:
                        {idea['content']}
                        """
                        improve_result = ask_gpt(prompt_improve, "gpt-4", 0.3)
                        st.write("**AI 추가 개선/분석 결과**:")
                        st.write(improve_result)

                st.write("---")

###############################################################################
# 메인 함수
###############################################################################
def main():
    st.title("studyhelper")

    st.warning("저작권에 유의해 파일을 업로드하세요.")
    st.info("ChatGPT는 실수를 할 수 있습니다. 중요한 정보를 반드시 추가 확인하세요.")

    # 사이드바에서 탭 구분
    tab = st.sidebar.radio("메뉴 선택", ("GPT 채팅", "DOCX 분석", "커뮤니티"))

    if tab == "GPT 채팅":
        st.subheader("GPT-4 채팅")
        chat_interface()

    elif tab == "DOCX 분석":
        st.subheader("문서 분석 (DOCX, PDF 파일 지원)")
        uploaded_file = st.file_uploader("DOCX 또는 PDF 파일을 업로드하세요", type=['docx', 'pdf'])

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            file_hash = hashlib.md5(file_bytes).hexdigest()

            # 새 파일인지 판별
            if ("uploaded_file_hash" not in st.session_state or
                st.session_state.uploaded_file_hash != file_hash):
                st.session_state.uploaded_file_hash = file_hash
                st.session_state.extracted_text = ""
                st.session_state.doc_analysis = {}
                st.session_state.processed = False

            # 아직 처리하지 않았다면 (업로드와 동시에 자동 진행)
            if not st.session_state.processed:
                raw_text = extract_text_from_file(uploaded_file)
                if raw_text.strip():
                    st.session_state.extracted_text = raw_text
                    st.success("파일 텍스트 추출 완료!")
                    with st.spinner("문서 분석 중 (요약, 중요 내용 추출, 질문 생성)..."):
                        analysis_result = analyze_document_text(raw_text)
                        st.session_state.doc_analysis = analysis_result
                else:
                    st.error("파일에서 텍스트를 추출할 수 없습니다.")
                st.session_state.processed = True

            # 결과 표시
            if st.session_state.get("processed", False):
                if st.session_state.extracted_text.strip():
                    st.write("## 추출된 문서 내용")
                    st.write(st.session_state.extracted_text)

                    if st.session_state.doc_analysis:
                        analysis_result = st.session_state.doc_analysis
                        st.write("## 요약")
                        st.write(analysis_result.get("summary", ""))
                        st.write("## 중요 내용")
                        st.write(analysis_result.get("important_content", ""))
                        st.write("## 생성된 질문")
                        st.write(analysis_result.get("questions", ""))
                else:
                    st.write("## 추출 결과를 표시할 수 없습니다.")

    else:
        community_investment_tab()

if __name__ == "__main__":
    main()
