import os
import openai
import streamlit as st
from dotenv import load_dotenv
import docx2txt

# PDF/PPTX 파싱용
import PyPDF2
from pptx import Presentation

# (HWP 파싱은 별도 라이브러리 필요하거나 변환 과정 필요)
# import hwp_parser ...

# ========================
# 환경변수 로드
# ========================
load_dotenv('.env')
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    st.warning("OPENAI_API_KEY가 설정되지 않았습니다.")
else:
    openai.api_key = openai_api_key

# ========================
# 커뮤니티 아이디어 모델
# ========================
class CommunityIdea:
    def __init__(self, title, content, auto_analysis="", likes=0, dislikes=0, investment=0,
                 comments=None, team_members=None):
        self.title = title
        self.content = content
        self.auto_analysis = auto_analysis  # GPT가 문서를 분석한 결과 or 자동분석 결과
        self.likes = likes
        self.dislikes = dislikes
        self.investment = investment
        self.comments = comments if comments else []
        self.team_members = team_members if team_members else []
        self.swot_analysis = ""
        self.customer_needs = ""

# ========================
# 세션 초기화
# ========================
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # GPT 채팅 히스토리

if "uploaded_text" not in st.session_state:
    st.session_state["uploaded_text"] = ""  # 업로드된 문서 텍스트

if "doc_analysis" not in st.session_state:
    st.session_state["doc_analysis"] = ""   # GPT 분석 결과

if "gpt_questions" not in st.session_state:
    st.session_state["gpt_questions"] = ""  # GPT가 자동으로 묻는 질문

if "community_ideas" not in st.session_state:
    st.session_state["community_ideas"] = [] # 커뮤니티 아이디어 목록

# ========================
# GPT 호출 함수
# ========================
def ask_gpt(prompt, max_tokens=200, temperature=0.7):
    if not openai.api_key:
        return "오류: OpenAI API 키가 없습니다."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"에러 발생: {e}"

# ========================
# 문서 파싱 함수 (PDF, PPTX, DOCX, HWP)
# ========================
def parse_file(uploaded_file):
    """파일 형식에 따라 텍스트를 추출"""
    filename = uploaded_file.name.lower()

    # 1) PDF
    if filename.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = []
        for page in pdf_reader.pages:
            text.append(page.extract_text())
        return "\n".join(text)

    # 2) PPT/PPTX
    elif filename.endswith(".ppt") or filename.endswith(".pptx"):
        # pptx만 python-pptx에서 안정적으로 처리
        prs = Presentation(uploaded_file)
        text = []
        for slide in prs.slides:
            slide_texts = []
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    slide_texts.append(shape.text)
            text.append("\n".join(slide_texts))
        return "\n".join(text)

    # 3) DOCX
    elif filename.endswith(".docx"):
        doc_text = docx2txt.process(uploaded_file)
        return doc_text if doc_text else ""

    # 4) HWP (별도 라이브러리 필요)
    elif filename.endswith(".hwp"):
        # 실제 파싱 라이브러리 사용 or hwp to pdf/docx 변환 로직 필요
        return "(HWP 파일은 아직 파싱이 구현되지 않았습니다. 별도 라이브러리 필요)"

    else:
        return "(지원하지 않는 파일 형식이거나 확장자가 없습니다)"

# ========================
# 메인 UI
# ========================
def main():
    st.title("StudyHelper")

    menu = st.sidebar.radio("메뉴", ["문서 업로드 & 자동 분석", "GPT 채팅", "커뮤니티"])
    if menu == "문서 업로드 & 자동 분석":
        run_file_analysis()
    elif menu == "GPT 채팅":
        run_gpt_chat()
    else:
        run_community_page()

# ========================
# 1) 문서 업로드 & 자동 분석
# ========================
def run_file_analysis():
    st.subheader("문서 업로드 & 자동 GPT 분석")

    uploaded_file = st.file_uploader("PDF/PPTX/DOCX/HWP 파일을 업로드하세요", type=["pdf","ppt","pptx","docx","hwp"])

    # 자동 분석
    if uploaded_file is not None:
        # 1) 파일 파싱
        with st.spinner("파일을 파싱 중..."):
            parsed_text = parse_file(uploaded_file)
        st.write("**추출된 텍스트**:")
        st.session_state["uploaded_text"] = parsed_text
        st.write(parsed_text)

        # 2) GPT 자동 분석
        with st.spinner("GPT 분석 중..."):
            doc_analysis = ask_gpt(f"다음 문서를 분석하고, 핵심내용을 요약 후 추가로 궁금해할 질문도 함께 제시해줘:\n{parsed_text}")
        st.session_state["doc_analysis"] = doc_analysis
        st.success("문서 자동 분석 완료")

        # GPT가 질문을 던지는 경우(문서 내부정보 기반)
        # (단순 예시. 실제로는 doc_analysis에서 "궁금한점:" 형태를 파싱하거나 추가 작업)
        st.write("### GPT 분석 결과")
        st.write(doc_analysis)

        # 사용자가 GPT가 던진 질문에 답할 수 있도록(간단히)
        st.write("GPT가 추가 질문을 했다면, 아래에 답변해볼 수 있습니다:")
        user_ans = st.text_input("GPT가 묻는 질문에 대한 나의 대답(옵션)")
        if st.button("GPT에게 답장하기"):
            with st.spinner("GPT에게 답변 전달 중..."):
                # GPT: 유저 답변을 추가 대화 context로
                followup_response = ask_gpt(
                    f"이전 문서 분석 결과: {doc_analysis}\n"
                    f"사용자가 질문에 이렇게 답했습니다: {user_ans}\n"
                    f"추가로 조언이나 정보를 제공해줘."
                )
                st.write("### 추가 정보")
                st.write(followup_response)

# ========================
# 2) GPT 채팅
# ========================
def run_gpt_chat():
    st.subheader("GPT 채팅")

    # 기존 대화 표시
    for chat_item in st.session_state["chat_history"]:
        role = chat_item["role"]
        msg = chat_item["message"]
        if role == "user":
            with st.chat_message("user"):
                st.write(msg)
        else:
            with st.chat_message("assistant"):
                st.write(msg)

    # 채팅 입력
    user_input = st.chat_input("메시지를 입력하세요:")
    if user_input:
        st.session_state["chat_history"].append({"role": "user", "message": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("GPT가 응답 중..."):
            gpt_response = ask_gpt(user_input)
            st.session_state["chat_history"].append({"role": "assistant", "message": gpt_response})

        with st.chat_message("assistant"):
            st.write(gpt_response)

# ========================
# 3) 커뮤니티
# ========================
def run_community_page():
    st.subheader("커뮤니티: 아이디어 공유 & 투자")

    # 아이디어 등록
    title = st.text_input("아이디어 제목")
    content = st.text_area("아이디어 내용")
    if st.button("아이디어 등록"):
        # 등록 시 GPT로 SWOT + 소비자 니즈 분석
        swot_prompt = f"다음 아이디어에 대해 간단한 SWOT(Strengths, Weaknesses, Opportunities, Threats) 분석:\n{content}"
        customer_prompt = f"다음 아이디어가 있다면, 고객(소비자) 니즈나 타겟, 시장 분석에서 주의할 점, 개선 방향:\n{content}"

        with st.spinner("자동 분석 중..."):
            swot_result = ask_gpt(swot_prompt)
            customer_result = ask_gpt(customer_prompt)

        new_idea = CommunityIdea(
            title=title,
            content=content,
            auto_analysis="자동분석(기본)",
        )
        new_idea.swot_analysis = swot_result
        new_idea.customer_needs = customer_result

        st.session_state["community_ideas"].append(new_idea)
        st.success("아이디어 등록 & 자동 SWOT/소비자 분석 완료!")

    st.write("---")
    st.write("### 아이디어 목록")
    ideas = st.session_state["community_ideas"]
    if not ideas:
        st.write("등록된 아이디어가 없습니다.")
        return

    for idx, idea in enumerate(ideas):
        with st.expander(f"{idx+1}. {idea.title}"):
            st.write(f"**내용**: {idea.content}")
            # 분석결과 표시
            if idea.swot_analysis:
                with st.expander("SWOT 분석 결과"):
                    st.write(idea.swot_analysis)
            if idea.customer_needs:
                with st.expander("고객(소비자) 니즈/분석 결과"):
                    st.write(idea.customer_needs)

            # 좋아요/싫어요/투자
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"👍 {idea.likes}")
                if st.button(f"좋아요_{idx}"):
                    idea.likes += 1
                    st.experimental_rerun()
            with col2:
                st.write(f"👎 {idea.dislikes}")
                if st.button(f"싫어요_{idx}"):
                    idea.dislikes += 1
                    st.experimental_rerun()
            with col3:
                st.write(f"💰 {idea.investment}")
                if st.button(f"투자 +100_{idx}"):
                    idea.investment += 100
                    st.experimental_rerun()

            # 팀원 합류
            st.write("### 팀원 목록")
            if not idea.team_members:
                st.write("아직 팀원이 없습니다.")
            else:
                for member in idea.team_members:
                    st.write(f"- {member}")

            if st.button(f"팀원 합류_{idx}"):
                idea.team_members.append("익명사용자")
                st.success("팀에 합류했습니다!")
                st.experimental_rerun()

            # 댓글
            st.write("### 댓글")
            if not idea.comments:
                st.write("댓글이 없습니다.")
            else:
                for cmt in idea.comments:
                    st.write(f"- {cmt}")

            new_comment = st.text_input(f"댓글 달기 (아이디어#{idx})", key=f"comment_{idx}")
            if st.button(f"댓글 등록_{idx}"):
                if new_comment.strip():
                    idea.comments.append(new_comment.strip())
                    st.experimental_rerun()
                else:
                    st.warning("댓글 내용을 입력하세요.")

            # 삭제 버튼
            if st.button(f"아이디어 삭제_{idx}", key=f"delete_{idx}"):
                st.session_state["community_ideas"].pop(idx)
                st.experimental_rerun()


# 실행
if __name__ == "__main__":
    main()
