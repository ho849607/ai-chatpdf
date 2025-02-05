import os
import json
import openai
import streamlit as st
from dotenv import load_dotenv
import docx2txt
import PyPDF2
from pptx import Presentation

# -------------------------
# 환경 변수 로드 (OpenAI API)
# -------------------------
load_dotenv('.env')
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    st.warning("OPENAI_API_KEY가 설정되지 않았습니다.")
else:
    openai.api_key = openai_api_key

# -------------------------
# CommunityIdea 모델 (커뮤니티 아이디어)
# -------------------------
class CommunityIdea:
    def __init__(
        self,
        title,
        content,
        auto_analysis="",
        likes=0,
        dislikes=0,
        investment=0,
        comments=None,
        team_members=None,
    ):
        self.title = title
        self.content = content
        self.auto_analysis = auto_analysis
        self.likes = likes
        self.dislikes = dislikes
        self.investment = investment
        self.comments = comments if comments else []
        self.team_members = team_members if team_members else []
        self.swot_analysis = ""
        self.customer_needs = ""
        # 추가: MERCE 및 BMC 분석 결과
        self.merce_analysis = ""
        self.bmc_analysis = ""

# -------------------------
# JSON 파일로 저장/로드
# -------------------------
IDEA_FILE = "ideas.json"

def load_ideas():
    if not os.path.exists(IDEA_FILE):
        return []
    try:
        with open(IDEA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        st.error(f"아이디어 파일 로드 중 오류 발생: {e}")
        return []
    ideas = []
    for item in data:
        idea = CommunityIdea(
            title=item["title"],
            content=item["content"],
            auto_analysis=item.get("auto_analysis", ""),
            likes=item.get("likes", 0),
            dislikes=item.get("dislikes", 0),
            investment=item.get("investment", 0),
            comments=item.get("comments", []),
            team_members=item.get("team_members", []),
        )
        idea.swot_analysis = item.get("swot_analysis", "")
        idea.customer_needs = item.get("customer_needs", "")
        idea.merce_analysis = item.get("merce_analysis", "")
        idea.bmc_analysis = item.get("bmc_analysis", "")
        ideas.append(idea)
    return ideas

def save_ideas(ideas):
    data = []
    for idea in ideas:
        data.append({
            "title": idea.title,
            "content": idea.content,
            "auto_analysis": idea.auto_analysis,
            "likes": idea.likes,
            "dislikes": idea.dislikes,
            "investment": idea.investment,
            "comments": idea.comments,
            "team_members": idea.team_members,
            "swot_analysis": idea.swot_analysis,
            "customer_needs": idea.customer_needs,
            "merce_analysis": idea.merce_analysis,
            "bmc_analysis": idea.bmc_analysis,
        })
    try:
        with open(IDEA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"아이디어 저장 중 오류 발생: {e}")

# -------------------------
# 세션 초기화
# -------------------------
# 매 페이지 로드시 파일에서 최신 아이디어를 불러오도록 수정
st.session_state["community_ideas"] = load_ideas()

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "uploaded_text" not in st.session_state:
    st.session_state["uploaded_text"] = ""

if "doc_analysis" not in st.session_state:
    st.session_state["doc_analysis"] = ""

if "extra_info" not in st.session_state:
    st.session_state["extra_info"] = ""

# 예시: 사용자 프로필 데이터 (실제 구현에서는 회원가입 정보를 별도 저장)
if "user_profile" not in st.session_state:
    st.session_state["user_profile"] = {
        "username": "익명사용자",
        "experience": "개발, 스타트업 참여 경험 있음",
        "preferences": "핀테크, AI, 블록체인"
    }

# -------------------------
# GPT 호출 함수 (max_tokens 늘림)
# -------------------------
def ask_gpt(prompt, max_tokens=600, temperature=0.7):
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

# -------------------------
# 파일 파싱 함수
# -------------------------
def parse_file(uploaded_file):
    filename = uploaded_file.name.lower()
    try:
        if filename.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = []
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
            return "\n".join(text)
        elif filename.endswith((".ppt", ".pptx")):
            prs = Presentation(uploaded_file)
            text = []
            for slide in prs.slides:
                slide_texts = []
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        slide_texts.append(shape.text)
                text.append("\n".join(slide_texts))
            return "\n".join(text)
        elif filename.endswith(".docx"):
            doc_text = docx2txt.process(uploaded_file)
            return doc_text if doc_text else ""
        elif filename.endswith(".hwp"):
            return "(HWP 파일은 아직 파싱이 구현되지 않았습니다.)"
        else:
            return "(지원하지 않는 파일 형식)"
    except Exception as e:
        return f"파일 파싱 중 오류 발생: {e}"

# -------------------------
# 팀원 추천 함수 (예시)
# -------------------------
def recommend_team_for_user(user_profile, idea):
    """
    사용자의 경력 및 선호 데이터를 바탕으로 아이디어와의 적합도를 평가하여 추천 메시지를 반환.
    실제 구현에서는 AI 분석 또는 규칙 기반 추천 알고리즘을 적용할 수 있음.
    """
    # 예시: 사용자의 선호에 "핀테크"가 포함되어 있고, 아이디어 내용에 "핀테크"가 언급되면 추천
    if "핀테크" in user_profile["preferences"] and "핀테크" in idea.content:
        return "이 아이디어는 귀하의 핀테크 선호와 잘 맞습니다!"
    else:
        return "더 다양한 팀 매칭 기회를 확인해보세요."

# -------------------------
# 메인 Streamlit 함수
# -------------------------
def main():
    st.title("StudyHelper")

    menu = st.sidebar.radio("메뉴", ["GPT 채팅", "문서 업로드 & 자동 분석", "커뮤니티"])

    if menu == "GPT 채팅":
        run_gpt_chat()
    elif menu == "문서 업로드 & 자동 분석":
        run_file_analysis()
    else:
        run_community_page()

# -------------------------
# 1) GPT 채팅
# -------------------------
def run_gpt_chat():
    st.subheader("GPT 채팅")

    for chat_item in st.session_state["chat_history"]:
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
            gpt_response = ask_gpt(user_input)
            st.session_state["chat_history"].append({"role": "assistant", "message": gpt_response})

        with st.chat_message("assistant"):
            st.write(gpt_response)

# -------------------------
# 2) 문서 업로드 & 자동 분석
# -------------------------
def run_file_analysis():
    st.subheader("문서 업로드 & 자동 GPT 분석")
    uploaded_file = st.file_uploader("PDF/PPTX/DOCX/HWP 파일을 업로드하세요", type=["pdf", "ppt", "pptx", "docx", "hwp"])

    if uploaded_file is not None:
        # 파일 파싱
        with st.spinner("파일 파싱 중..."):
            parsed_text = parse_file(uploaded_file)
        st.write("**추출된 텍스트**:")
        st.session_state["uploaded_text"] = parsed_text
        st.write(parsed_text)

        # GPT 분석 (문서 요약 및 질문)
        with st.spinner("GPT 분석(요약+질문) 중..."):
            doc_analysis = ask_gpt(
                f"다음 문서를 분석하고, 핵심내용을 요약한 후 추가로 궁금해할 질문을 제시해줘:\n{parsed_text}"
            )
        st.session_state["doc_analysis"] = doc_analysis

        # 추가 정보 제공 (배경지식, 유사사례 등)
        with st.spinner("GPT가 추가 정보 파악 중..."):
            extra_info = ask_gpt(
                f"문서 내용: {parsed_text}\n\n"
                f"요약+질문: {doc_analysis}\n\n"
                f"이 문서를 살펴보는 사람이 관심 있어 할 만한 배경지식, 관련 사례, 추가 팁 등을 제공해줘."
            )
        st.session_state["extra_info"] = extra_info

        st.success("문서 자동 분석 및 추가 정보 제공 완료")

        st.write("### GPT 분석 결과")
        st.write(doc_analysis)

        st.write("### GPT 추가 정보")
        st.write(extra_info)

        # 사용자가 GPT 질문에 답변할 수 있도록 입력받음
        user_ans = st.text_input("GPT가 궁금해하는 질문에 대한 답변(옵션)", key="doc_user_ans")
        if st.button("GPT에게 답장하기", key="reply_button"):
            with st.spinner("GPT에게 답변 전달 중..."):
                followup = ask_gpt(
                    f"문서 분석 결과: {doc_analysis}\n"
                    f"추가 정보: {extra_info}\n"
                    f"사용자가 질문에 이렇게 답했습니다: {user_ans}\n"
                    f"추가 조언이나 정보를 제공해줘."
                )
                st.write("### 후속 정보")
                st.write(followup)

# -------------------------
# 3) 커뮤니티
# -------------------------
def run_community_page():
    st.subheader("커뮤니티: 아이디어 공유 & 투자")

    with st.form(key="idea_form", clear_on_submit=True):
        title = st.text_input("아이디어 제목")
        content = st.text_area("아이디어 내용")
        submitted = st.form_submit_button("아이디어 등록")
    if submitted and title.strip() and content.strip():
        # GPT로 SWOT, 고객 분석, MERCE 및 BMC 진행
        swot_prompt = f"다음 아이디어에 대해 간단한 SWOT 분석을 해줘:\n{content}"
        customer_prompt = f"이 아이디어에 대한 고객(소비자) 니즈나 시장분석 요약을 해줘:\n{content}"
        merce_prompt = f"이 아이디어에 대해 MERCE 분석을 해줘:\n{content}"
        bmc_prompt = f"이 아이디어에 대해 비즈니스 모델 캔버스(BMC)를 정리해줘:\n{content}"
        with st.spinner("자동 분석 중..."):
            swot_result = ask_gpt(swot_prompt)
            customer_result = ask_gpt(customer_prompt)
            merce_result = ask_gpt(merce_prompt)
            bmc_result = ask_gpt(bmc_prompt)
        
        new_idea = CommunityIdea(
            title=title,
            content=content,
            auto_analysis="자동분석(기본)"
        )
        new_idea.swot_analysis = swot_result
        new_idea.customer_needs = customer_result
        new_idea.merce_analysis = merce_result
        new_idea.bmc_analysis = bmc_result

        # 파일에 저장된 기존 아이디어와 합쳐서 저장
        ideas = load_ideas()
        ideas.append(new_idea)
        st.session_state["community_ideas"] = ideas
        save_ideas(ideas)
        st.success("아이디어 등록 및 자동 분석 완료!")

    st.write("---")
    st.write("### 아이디어 목록")
    ideas = st.session_state["community_ideas"]
    if not ideas:
        st.write("등록된 아이디어가 없습니다.")
        return

    for idx, idea in enumerate(ideas):
        with st.expander(f"{idx+1}. {idea.title}", expanded=False):
            st.write(f"**내용**: {idea.content}")

            if idea.swot_analysis:
                st.markdown("**SWOT 분석 결과:**")
                st.write(idea.swot_analysis)
            if idea.merce_analysis:
                st.markdown("**MERCE 분석 결과:**")
                st.write(idea.merce_analysis)
            if idea.bmc_analysis:
                st.markdown("**BMC 분석 결과:**")
                st.write(idea.bmc_analysis)
            if idea.customer_needs:
                st.markdown("**고객(소비자) 분석:**")
                st.write(idea.customer_needs)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"👍 {idea.likes}")
                if st.button("좋아요", key=f"like_{idx}"):
                    idea.likes += 1
                    save_ideas(ideas)
                    st.experimental_rerun()
            with col2:
                st.write(f"👎 {idea.dislikes}")
                if st.button("싫어요", key=f"dislike_{idx}"):
                    idea.dislikes += 1
                    save_ideas(ideas)
                    st.experimental_rerun()
            with col3:
                st.write(f"💰 {idea.investment}")
                if st.button("투자 +100", key=f"invest_{idx}"):
                    idea.investment += 100
                    save_ideas(ideas)
                    st.experimental_rerun()

            st.write("### 팀원 목록")
            if not idea.team_members:
                st.write("아직 팀원이 없습니다.")
            else:
                for member in idea.team_members:
                    st.write(f"- {member}")
            if st.button("팀원 합류", key=f"join_{idx}"):
                # 사용자 프로필과 아이디어를 기반으로 팀 매칭 추천 (예시)
                recommendation = recommend_team_for_user(st.session_state["user_profile"], idea)
                idea.team_members.append(st.session_state["user_profile"]["username"])
                save_ideas(ideas)
                st.success(f"팀에 합류했습니다! 추천: {recommendation}")
                st.experimental_rerun()

            st.write("### 댓글")
            if not idea.comments:
                st.write("댓글이 없습니다.")
            else:
                for comment in idea.comments:
                    st.write(f"- {comment}")

            new_comment = st.text_input("댓글 달기", key=f"comment_{idx}")
            if st.button("댓글 등록", key=f"submit_comment_{idx}"):
                if new_comment.strip():
                    idea.comments.append(new_comment.strip())
                    save_ideas(ideas)
                    st.experimental_rerun()
                else:
                    st.warning("댓글 내용을 입력하세요.")

            if st.button("아이디어 삭제", key=f"delete_{idx}"):
                ideas.pop(idx)
                st.session_state["community_ideas"] = ideas
                save_ideas(ideas)
                st.experimental_rerun()

if __name__ == "__main__":
    main()
