import os
import json
import streamlit as st

# -------------------------
# 아이디어 모델
# -------------------------
class CommunityIdea:
    def __init__(self, title, content, likes=0, dislikes=0, investment=0,
                 comments=None, team_members=None, auto_analysis=""):
        self.title = title
        self.content = content
        self.likes = likes
        self.dislikes = dislikes
        self.investment = investment
        self.comments = comments if comments else []
        self.team_members = team_members if team_members else []
        self.auto_analysis = auto_analysis

# -------------------------
# 아이디어 저장/불러오기
# -------------------------
IDEA_FILE = "ideas.json"

def load_ideas():
    """ideas.json 파일이 있으면 로드하여 CommunityIdea 리스트 반환"""
    if not os.path.exists(IDEA_FILE):
        return []
    with open(IDEA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    ideas = []
    for item in data:
        # json 구조를 CommunityIdea로 복원
        idea = CommunityIdea(
            title=item["title"],
            content=item["content"],
            likes=item.get("likes", 0),
            dislikes=item.get("dislikes", 0),
            investment=item.get("investment", 0),
            comments=item.get("comments", []),
            team_members=item.get("team_members", []),
            auto_analysis=item.get("auto_analysis", "")
        )
        ideas.append(idea)
    return ideas

def save_ideas(ideas):
    """CommunityIdea 리스트를 json 파일로 저장"""
    data = []
    for idea in ideas:
        data.append({
            "title": idea.title,
            "content": idea.content,
            "likes": idea.likes,
            "dislikes": idea.dislikes,
            "investment": idea.investment,
            "comments": idea.comments,
            "team_members": idea.team_members,
            "auto_analysis": idea.auto_analysis
        })
    with open(IDEA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# -------------------------
# Streamlit 메인
# -------------------------
def main():
    st.title("아이디어 등록 예시 with 파일 저장")

    # 세션에 아이디어 리스트가 없으면 파일에서 로드
    if "community_ideas" not in st.session_state:
        st.session_state["community_ideas"] = load_ideas()

    # 아이디어 입력
    title = st.text_input("아이디어 제목")
    content = st.text_area("아이디어 내용")

    if st.button("아이디어 등록"):
        # 빈값 체크
        if not title.strip() or not content.strip():
            st.warning("제목과 내용을 모두 입력하세요.")
        else:
            # 새 아이디어 생성
            new_idea = CommunityIdea(title=title, content=content)
            # 세션 아이디어 목록에 추가
            st.session_state["community_ideas"].append(new_idea)
            # 파일로 저장
            save_ideas(st.session_state["community_ideas"])
            st.success("아이디어가 등록되었습니다!")

    st.write("---")
    st.write("## 아이디어 목록")

    # 등록된 아이디어 표시
    ideas = st.session_state["community_ideas"]
    if not ideas:
        st.write("등록된 아이디어가 없습니다.")
    else:
        for idx, idea in enumerate(ideas):
            with st.expander(f"{idx+1}. {idea.title}"):
                st.write(f"**내용**: {idea.content}")
                st.write(f"👍 좋아요: {idea.likes}")
                st.write(f"👎 싫어요: {idea.dislikes}")
                st.write(f"💰 투자: {idea.investment}")
                
                # 팀원
                st.write("### 팀원 목록")
                if not idea.team_members:
                    st.write("아직 팀원이 없습니다.")
                else:
                    for m in idea.team_members:
                        st.write(f"- {m}")
                # 팀원 합류 버튼 (예시)
                if st.button(f"팀원 합류_{idx}"):
                    idea.team_members.append("익명사용자")
                    # 저장
                    save_ideas(st.session_state["community_ideas"])
                    st.experimental_rerun()

                # 댓글
                st.write("### 댓글 목록")
                if not idea.comments:
                    st.write("아직 댓글이 없습니다.")
                else:
                    for cmt in idea.comments:
                        st.write(f"- {cmt}")

                new_comment = st.text_input(f"댓글 달기 (#{idx})", key=f"comment_{idx}")
                if st.button(f"댓글 등록_{idx}"):
                    if new_comment.strip():
                        idea.comments.append(new_comment.strip())
                        save_ideas(st.session_state["community_ideas"])
                        st.experimental_rerun()
                    else:
                        st.warning("댓글 내용을 입력하세요.")

                # 좋아요 / 싫어요 / 투자 / 삭제
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if st.button(f"좋아요_{idx}", key=f"like_{idx}"):
                        idea.likes += 1
                        save_ideas(st.session_state["community_ideas"])
                        st.experimental_rerun()
                with col2:
                    if st.button(f"싫어요_{idx}", key=f"dislike_{idx}"):
                        idea.dislikes += 1
                        save_ideas(st.session_state["community_ideas"])
                        st.experimental_rerun()
                with col3:
                    if st.button(f"투자+100_{idx}", key=f"invest_{idx}"):
                        idea.investment += 100
                        save_ideas(st.session_state["community_ideas"])
                        st.experimental_rerun()
                with col4:
                    if st.button(f"삭제_{idx}", key=f"delete_{idx}"):
                        st.session_state["community_ideas"].pop(idx)
                        save_ideas(st.session_state["community_ideas"])
                        st.experimental_rerun()

if __name__ == "__main__":
    main()
