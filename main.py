import os
import streamlit as st
from io import BytesIO
from dotenv import load_dotenv
import openai
from pathlib import Path
import hashlib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# docx2txt 설치 확인
try:
    import docx2txt
    DOCX_ENABLED = True
except ImportError:
    DOCX_ENABLED = False

# 초기 NLTK 다운로드(존재하지 않을 때만)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

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

# Streamlit 페이지 설정
st.set_page_config(page_title="studyhelper")

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

openai.api_key = openai_api_key

###############################################################################
# 고급 분석(Chunk 분할 + 중요도 평가) 함수
###############################################################################
def chunk_text_by_heading(docx_text):
    """
    [데모용 함수]
    docx_text 안에서 '===Heading:'이라는 인위적 라벨을 기준으로 Chunk를 나눕니다.
    
    실제 Word 문서의 Heading(제목1, 제목2 등)을 활용하려면 python-docx 등의 라이브러리로
    paragraph.style.name을 확인하여 분할하는 방식을 권장합니다.
    """
    lines = docx_text.split('\n')
    chunks = []
    current_chunk = []
    heading_title = "NoHeading"
    chunk_id = 0

    for line in lines:
        if line.strip().startswith("===Heading:"):
            # 기존 chunk 저장
            if current_chunk:
                chunks.append({
                    "id": chunk_id,
                    "heading": heading_title,
                    "text": "\n".join(current_chunk)
                })
                chunk_id += 1
                current_chunk = []
            heading_title = line.replace("===Heading:", "").strip()
        else:
            current_chunk.append(line)

    # 마지막 chunk 처리
    if current_chunk:
        chunks.append({
            "id": chunk_id,
            "heading": heading_title,
            "text": "\n".join(current_chunk)
        })
    return chunks

def gpt_evaluate_importance(chunk_text, language='korean'):
    """
    이 함수는 GPT를 이용해:
      1) Chunk의 '중요도'를 1~5 사이 정수로 평가
      2) 한두 문장 요약
    을 수행하는 간단 예시입니다.
    
    실제 운영 환경에서는 Prompt/파싱 로직을 더 견고하게 설계해 주세요.
    """
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage

    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    
    if language == 'korean':
        prompt = f"""
        아래 텍스트가 있습니다. 이 텍스트가 전체 문서에서 얼마나 중요한지 1~5 사이 정수로 결정하고,
        한두 문장으로 요약해 주세요.

        텍스트:
        {chunk_text}

        응답 형식 예시:
        중요도: 4
        요약: ~~
        """
    else:
        prompt = f"""
        The following text is given. Please determine how important it is (1 to 5), 
        and provide a one or two-sentence summary.

        Text:
        {chunk_text}

        Example response format:
        Importance: 4
        Summary: ...
        """

    messages = [HumanMessage(content=prompt)]
    response = llm(messages).content.strip()

    importance = 3  # 기본값
    short_summary = ""
    for line in response.split('\n'):
        if "중요도:" in line or "Importance:" in line:
            try:
                number_str = line.split(':')[-1].strip()
                importance = int(number_str)
            except:
                pass
        if "요약:" in line or "Summary:" in line:
            short_summary = line.split(':', 1)[-1].strip()

    return importance, short_summary

def docx_advanced_processing(docx_text, language='korean'):
    """
    1) 문단/heading 단위로 chunk 분할 (현재는 '===Heading:' 문자열을 통해 인위적 분할)
    2) GPT로 각 chunk 중요도/간단 요약 평가
    3) chunk별 결과를 합쳐서 최종 문자열로 반환
    
    장점:
      - 문서가 길어도 chunk별로 나누어 처리 가능 -> GPT 토큰 비용 절감
      - chunk별 중요도 표시 -> 어떤 부분을 우선적으로 학습할지 한눈에 파악 가능
      - 문서 구조(Heading) 기반 접근 -> Heading별 요약을 별도로 확인 가능
    """
    chunks = chunk_text_by_heading(docx_text)
    combined_result = []

    for c in chunks:
        importance, short_summary = gpt_evaluate_importance(c["text"], language=language)
        c["importance"] = importance
        c["short_summary"] = short_summary
        combined_result.append(c)

    # chunk별 요약 내용을 합침
    final_summary_parts = []
    for c in combined_result:
        part = (
            f"=== [Chunk #{c['id']}] Heading: {c['heading']} ===\n"
            f"중요도: {c['importance']}\n"
            f"요약: {c['short_summary']}\n"
            f"원문 일부:\n{c['text'][:200]}...\n"  # 필요 시 원문 일부만 표시
        )
        final_summary_parts.append(part)

    final_summary = "\n".join(final_summary_parts)
    return final_summary

###############################################################################
# 채팅 및 GPT 관련 함수
###############################################################################
def add_chat_message(role, message):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({"role": role, "message": message})

def ask_gpt_question(question, language='korean'):
    """
    langchain을 사용한 GPT-4 질의 응답
    """
    from langchain.chat_models import ChatOpenAI
    from langchain.callbacks import StreamingStdOutCallbackHandler
    from langchain.schema import HumanMessage

    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )

    if language == 'korean':
        prompt = f"다음 질문에 답변해 주세요:\n\n{question}"
    else:
        prompt = f"Please answer the following question:\n\n{question}"

    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    return response.content

def chat_interface():
    """
    사용자와 GPT-4의 대화 인터페이스.
    """
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

    # 입력 필드
    user_chat_input = st.chat_input("메시지를 입력하세요:")
    if user_chat_input:
        # 사용자 메시지 저장
        add_chat_message("user", user_chat_input)
        with st.chat_message("user"):
            st.write(user_chat_input)

        # GPT 응답
        with st.spinner("GPT가 응답 중입니다..."):
            gpt_response = ask_gpt_question(user_chat_input, 'korean')
            add_chat_message("assistant", gpt_response)
            with st.chat_message("assistant"):
                st.write(gpt_response)

###############################################################################
# DOCX 텍스트 추출 함수
###############################################################################
def docx_to_text(upload_file):
    """
    docx2txt로 DOCX 파일의 텍스트를 추출합니다.
    """
    if not DOCX_ENABLED:
        st.warning("docx2txt가 설치되어 있지 않아 .docx 파일을 처리할 수 없습니다.")
        return ""
    try:
        text = docx2txt.process(BytesIO(upload_file.getvalue()))
        return text if text else ""
    except Exception as e:
        st.error(f"DOCX 파일 처리 중 오류가 발생했습니다: {e}")
        return ""

###############################################################################
# 메인 함수
###############################################################################
def main():
    # 페이지 상단 타이틀
    st.title("studyhelper")
    
    # 주의 문구
    st.warning("저작권에 유의해 파일을 업로드하세요.")
    st.info("ChatGPT는 실수를 할 수 있습니다. 중요한 정보를 반드시 추가 확인하세요.")

    # 채팅 인터페이스
    st.write("---")
    st.subheader("GPT-4 채팅")
    chat_interface()

    # 문서 업로드 섹션 (.docx 전용)
    st.write("---")
    st.subheader("DOCX 문서 분석 (고급 Chunk 단위 분석)")
    uploaded_file = st.file_uploader(
        "DOCX 파일을 업로드하세요 (문서 내에 '===Heading:'이라는 구분자를 추가해보세요!)",
        type=['docx']
    )

    if uploaded_file is not None:
        filename = uploaded_file.name
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()

        # 새 파일이 업로드되면 세션 상태 초기화
        if ("uploaded_file_hash" not in st.session_state or
            st.session_state.uploaded_file_hash != file_hash):
            st.session_state.uploaded_file_hash = file_hash
            st.session_state.extracted_text = ""
            st.session_state.summary = ""
            st.session_state.processed = False

        # 아직 처리 안된 상태라면, 문서 텍스트 추출 후 고급 분석
        if not st.session_state.processed:
            raw_text = docx_to_text(uploaded_file)
            if raw_text.strip():
                # 고급 분석 (chunk 분할 + 중요도/요약)
                with st.spinner("문서 고급 분석 진행 중..."):
                    advanced_summary = docx_advanced_processing(raw_text, language='korean')
                    st.session_state.summary = advanced_summary
                    st.session_state.extracted_text = raw_text
                    st.success("DOCX 고급 분석 완료!")
            else:
                st.error("DOCX에서 텍스트를 추출할 수 없습니다.")
                st.session_state.summary = ""
            
            st.session_state.processed = True

        # 결과 표시
        if st.session_state.get("processed", False):
            if 'summary' in st.session_state and st.session_state.summary.strip():
                st.write("## (고급) Chunk 기반 요약 & 중요도 결과")
                st.write(st.session_state.summary)
            else:
                st.write("## 요약 결과를 표시할 수 없습니다.")

if __name__ == "__main__":
    main()
