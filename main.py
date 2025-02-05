import os
import openai
import streamlit as st
from dotenv import load_dotenv
import json

# ========================
# .env 로드 (OPENAI_API_KEY 등)
# ========================
load_dotenv('.env')
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    st.warning("OPENAI_API_KEY가 설정되지 않았습니다.")
else:
    openai.api_key = openai_api_key

# ========================
# 데이터 모델 대체 (간단 dict/리스트)
# ========================
class CommunityIdea:
    def __init__(self, title, content, auto_analysis=None, likes=0, dislikes=0, investment=0, comments=None, team_members=None):
        self.title = title
        self.content = content
        self.auto_analysis = auto_analysis
        self.likes = likes
        self.dislikes = dislikes
        self.investment = investment
        self.comments = comments if comments else []
        self.team_members = team_members if team_members else []

# 세션 스테이트로 데이터 보관
if "community_ideas" not in st.session_state:
    st.session_state["community_ideas"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# ========================
# GPT 채팅 함수
# ========================
def chat_with_gpt(user_text):
    if not openai.api_key:
        return "오류: OpenAI API 키가 없습니다."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": user_text}
            ],
            max_tokens=150,
            temperature=0.7,
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"에러 발생: {e}"

# ========================
# Streamlit UI
# ========================
def main():
    st.title("StudyHelper (Python 버전)")

    menu = st.sidebar.radio("메뉴", ["GPT 채팅", "DOCX 분석", "커뮤니티"])

    if menu == "GPT 채팅":
        run_gpt_chat()
    elif menu == "DOCX 분석":
        run_docx_analysis()
    else:
        run_community_page()

# ========================
# GPT 채팅 화면
# ========================
def run_gpt_chat():
    st.subheader("GPT 채팅")
    # 기존 채팅 이력 표시
    for idx, chat_item in enumerate(st.session_state["chat_history"]):
        role = chat_item["role"]
        msg = chat_item["message"]
        if role == "user":
            with st.chat_message("user"):
                st.write(msg)
        else:
            with st.chat_message("assistant"):
                st.write(msg)

    user_input = st.chat_input("메시지를 입력하세요:")
    if user_input:
        st.session_state["chat_history"].append({"role": "user", "message": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("GPT가 응답 중..."):
            gpt_response = chat_with_gpt(user_input)
            st.session_state["chat_history"].append({"role": "assistant", "message": gpt_response})
        with st.chat_message("assistant"):
            st.write(gpt_response)

# ========================
# DOCX 분석 화면 (간단 예시)
# ========================
def run_docx_analysis():
    st.subheader("DOCX 문서 분석 (간단 예시)")

    # 텍스트 업로드 (실제로는 파일 업로드와 docx2txt 파싱 사용)
    user_text = st.text_area("여기에 DOCX 텍스트를 붙여넣으세요.")
    if st.button("텍스트 추출(가정)"):
        st.session_state["extracted_text"] = user_text
        st.success("DOCX 텍스트 추출 완료")

    if "extracted_text" in st.session_state and st.session_state["extracted_text"]:
        st.write("### 추출된 문서 내용")
        st.write(st.session_state["extracted_text"])

        if st.button("GPT로 분석하기 (예시)"):
            with st.spinner("GPT 분석 중..."):
                gpt_result = chat_with_gpt(f"문서를 분석하고 요약해줘:\n{st.session_state['extracted_text']}")
                st.session_state["docx_analysis_result"] = gpt_result

        if "docx_analysis_result" in st.session_state and st.session_state["docx_analysis_result"]:
            st.write("### GPT 분석 결과")
            st.write(st.session_state["docx_analysis_result"])

# ========================
# 커뮤니티 화면
# ========================
def run_community_page():
    st.subheader("아이디어 공유 & 투자 커뮤니티")

    # 새 아이디어 입력
    title = st.text_input("아이디어 제목")
    content = st.text_area("아이디어 내용 (간략 소개)")
    if st.button("아이디어 등록"):
        new_idea = CommunityIdea(title=title, content=content, auto_analysis="AI 자동분석 예시...")
        st.session_state["community_ideas"].append(new_idea)
        st.success("아이디어 등록 완료")

    st.write("---")
    st.write("## 아이디어 목록")

    ideas = st.session_state["community_ideas"]
    if not ideas:
        st.write("등록된 아이디어가 없습니다.")
        return

    for idx, idea in enumerate(ideas):
        with st.expander(f"{idx+1}. {idea.title}", expanded=False):
            st.write(f"**내용**: {idea.content}")
            if idea.autoAnalysis:
                st.write(f"**AI 분석**: {idea.autoAnalysis}")

            # 좋아요 / 싫어요 / 투자
            col1, col2, col3, col4 = st.columns([1,1,1,1])
            with col1:
                st.write(f"👍 {idea.likes}")
                if st.button(f"좋아요({idx})"):
                    idea.likes += 1
                    st.experimental_rerun()
            with col2:
                st.write(f"👎 {idea.dislikes}")
                if st.button(f"싫어요({idx})"):
                    idea.dislikes += 1
                    st.experimental_rerun()
            with col3:
                st.write(f"💰 {idea.investment}")
                if st.button(f"투자 +100({idx})"):
                    idea.investment += 100
                    st.experimental_rerun()
            with col4:
                if st.button(f"삭제({idx})"):
                    st.session_state["community_ideas"].pop(idx)
                    st.experimental_rerun()

            # 팀원 섹션
            st.write("### 팀원 합류")
            st.write(f"현재 팀원: {idea.teamMembers}") if idea.teamMembers else st.write("아직 팀원이 없습니다.")
            if st.button(f"팀원 합류 (아이디어#{idx})"):
                idea.teamMembers.append("익명사용자")
                st.success("팀에 합류했습니다!")
                st.experimental_rerun()

            # 댓글 표시
            st.write("### 댓글")
            if not idea.comments:
                st.write("아직 댓글이 없습니다.")
            else:
                for cmt in idea.comments:
                    st.write(f"- {cmt}")

            new_comment = st.text_input(f"댓글 달기 (아이디어#{idx})", key=f"comment_{idx}")
            if st.button(f"등록 (아이디어#{idx})"):
                if new_comment.strip():
                    idea.comments.append(new_comment.strip())
                    st.experimental_rerun()
                else:
                    st.warning("댓글 내용을 입력하세요")

# 실행
if __name__ == "__main__":
    main()
