import os
import json
import openai
import streamlit as st
from dotenv import load_dotenv
import docx2txt
import PyPDF2
from pptx import Presentation
import hashlib
import time

# -------------------------
# 블록체인 구현 (전자책 정보 위변조 방지)
# -------------------------
class Block:
    def __init__(self, index, timestamp, data, previous_hash, nonce=0):
        """
        :param index: 블록 번호
        :param timestamp: 블록 생성 시각 (초 단위 타임스탬프)
        :param data: 블록에 저장할 데이터 (예: 전자책 정보)
        :param previous_hash: 이전 블록의 해시값
        :param nonce: 채굴용 임의 값 (초기값 0)
        """
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        """
        블록의 내용을 JSON으로 변환하여 SHA-256 해시값 계산
        """
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce
        }, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def mine_block(self, difficulty):
        """
        채굴: 해시 앞부분에 '0'이 difficulty 수만큼 붙을 때까지 nonce를 증가시키며 해시값 재계산
        """
        target = "0" * difficulty
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()
        print(f"Block {self.index} mined: {self.hash}")

class Blockchain:
    def __init__(self, difficulty=2):
        self.chain = [self.create_genesis_block()]
        self.difficulty = difficulty  # 채굴 난이도

    def create_genesis_block(self):
        """
        최초의 블록(제네시스 블록) 생성
        """
        return Block(0, time.time(), "Genesis Block", "0")

    def get_latest_block(self):
        return self.chain[-1]

    def add_block(self, data):
        """
        새로운 데이터를 포함하는 블록을 생성하고 채굴하여 블록체인에 추가
        """
        new_index = len(self.chain)
        new_block = Block(new_index, time.time(), data, self.get_latest_block().hash)
        new_block.mine_block(self.difficulty)
        self.chain.append(new_block)

    def is_chain_valid(self):
        """
        체인 전체를 순회하며 각 블록의 해시가 올바른지, 이전 해시와 일치하는지 확인
        """
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            if current_block.hash != current_block.calculate_hash():
                print(f"Block {current_block.index}의 해시값이 일치하지 않습니다!")
                return False
            if current_block.previous_hash != previous_block.hash:
                print(f"Block {current_block.index}의 이전 해시가 올바르지 않습니다!")
                return False
        return True

# 전역 블록체인 인스턴스 (전자책 기록용)
idea_blockchain = Blockchain(difficulty=2)

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
# 전자책 모델
# -------------------------
class EBook:
    def __init__(self, title, description, purchase_price, rental_price, auto_analysis="", file_text="", review_analysis="", comments=None, purchase_count=0, rental_count=0):
        self.title = title
        self.description = description
        self.purchase_price = purchase_price
        self.rental_price = rental_price
        self.auto_analysis = auto_analysis
        self.file_text = file_text
        self.review_analysis = review_analysis
        self.comments = comments if comments else []
        self.purchase_count = purchase_count
        self.rental_count = rental_count

# -------------------------
# JSON 파일로 전자책 저장/로드
# -------------------------
EBOOK_FILE = "ebooks.json"

def load_ebooks():
    if not os.path.exists(EBOOK_FILE):
        return []
    try:
        with open(EBOOK_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        st.error(f"전자책 파일 로드 중 오류 발생: {e}")
        return []
    ebooks = []
    for item in data:
        ebook = EBook(
            title=item["title"],
            description=item["description"],
            purchase_price=item.get("purchase_price", 0),
            rental_price=item.get("rental_price", 0),
            auto_analysis=item.get("auto_analysis", ""),
            file_text=item.get("file_text", ""),
            review_analysis=item.get("review_analysis", ""),
            comments=item.get("comments", []),
            purchase_count=item.get("purchase_count", 0),
            rental_count=item.get("rental_count", 0)
        )
        ebooks.append(ebook)
    return ebooks

def save_ebooks(ebooks):
    data = []
    for ebook in ebooks:
        data.append({
            "title": ebook.title,
            "description": ebook.description,
            "purchase_price": ebook.purchase_price,
            "rental_price": ebook.rental_price,
            "auto_analysis": ebook.auto_analysis,
            "file_text": ebook.file_text,
            "review_analysis": ebook.review_analysis,
            "comments": ebook.comments,
            "purchase_count": ebook.purchase_count,
            "rental_count": ebook.rental_count
        })
    try:
        with open(EBOOK_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"전자책 저장 중 오류 발생: {e}")

# -------------------------
# 세션 초기화 (기존 값이 없을 경우에만 초기화)
# -------------------------
if "ebooks" not in st.session_state:
    st.session_state["ebooks"] = load_ebooks()

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "uploaded_text" not in st.session_state:
    st.session_state["uploaded_text"] = ""

if "doc_analysis" not in st.session_state:
    st.session_state["doc_analysis"] = ""

if "extra_info" not in st.session_state:
    st.session_state["extra_info"] = ""

if "user_profile" not in st.session_state:
    st.session_state["user_profile"] = {
        "username": "익명사용자",
        "experience": "개발, 스타트업 참여 경험 있음",
        "preferences": "핀테크, AI, 블록체인",
        "membership": False  # 기본적으로 일반 회원
    }

# -------------------------
# GPT 호출 함수
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
# 메인 Streamlit 함수
# -------------------------
def main():
    st.title("StudyHelper - 전자책 플랫폼")

    # 좌측 사이드바에서 멤버십 상태 선택 (일반 회원/멤버십 가입)
    membership_status = st.sidebar.radio("멤버십 상태", ["일반 회원", "멤버십 가입"])
    st.session_state["user_profile"]["membership"] = (membership_status == "멤버십 가입")

    menu = st.sidebar.radio("메뉴", ["GPT 채팅", "전자책 업로드 & 자동 분석", "전자책 등록 & 구매/대여"])

    if menu == "GPT 채팅":
        run_gpt_chat()
    elif menu == "전자책 업로드 & 자동 분석":
        run_file_analysis()
    else:
        run_ebook_marketplace()

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
# 2) 전자책 업로드 & 자동 분석
# -------------------------
def run_file_analysis():
    st.subheader("전자책 업로드 & 자동 분석")
    uploaded_file = st.file_uploader("PDF/PPTX/DOCX/HWP 파일을 업로드하세요", type=["pdf", "ppt", "pptx", "docx", "hwp"])

    if uploaded_file is not None:
        with st.spinner("파일 파싱 중..."):
            parsed_text = parse_file(uploaded_file)
        st.write("**추출된 전자책 텍스트**:")
        st.session_state["uploaded_text"] = parsed_text
        st.write(parsed_text)

        with st.spinner("GPT 분석(요약 및 개선점) 중..."):
            doc_analysis = ask_gpt(
                f"다음 전자책 내용을 요약하고, 핵심 개선점과 중요한 부분을 알려줘:\n{parsed_text}"
            )
        st.session_state["doc_analysis"] = doc_analysis

        with st.spinner("GPT 추가 분석 중..."):
            extra_info = ask_gpt(
                f"전자책 내용: {parsed_text}\n\n"
                f"요약 및 개선점: {doc_analysis}\n\n"
                f"이 전자책에 대해 추가로 참고할 만한 배경지식, 사례, 팁 등을 제공해줘."
            )
        st.session_state["extra_info"] = extra_info

        st.success("전자책 자동 분석 및 추가 정보 제공 완료")

        st.write("### GPT 분석 결과")
        st.write(doc_analysis)

        st.write("### GPT 추가 정보")
        st.write(extra_info)

        user_ans = st.text_input("GPT가 제시한 질문에 대한 답변 (옵션)", key="doc_user_ans")
        if st.button("GPT에게 답장하기", key="reply_button"):
            with st.spinner("GPT에게 답변 전달 중..."):
                followup = ask_gpt(
                    f"전자책 분석 결과: {doc_analysis}\n"
                    f"추가 정보: {extra_info}\n"
                    f"사용자가 질문에 이렇게 답했습니다: {user_ans}\n"
                    f"추가 조언이나 정보를 제공해줘."
                )
                st.write("### 후속 정보")
                st.write(followup)

# -------------------------
# 3) 전자책 등록 & 구매/대여 (커뮤니티)
# -------------------------
def run_ebook_marketplace():
    st.subheader("전자책 등록 & 구매/대여")

    # 전자책 등록 폼
    with st.form(key="ebook_form", clear_on_submit=True):
        title = st.text_input("전자책 제목")
        description = st.text_area("전자책 설명")
        purchase_price = st.number_input("구매 가격 (원)", min_value=0, value=1000)
        rental_price = st.number_input("대여 가격 (원)", min_value=0, value=500)
        uploaded_file = st.file_uploader("전자책 파일 업로드 (PDF/PPTX/DOCX/HWP)", type=["pdf", "ppt", "pptx", "docx", "hwp"])
        submitted = st.form_submit_button("전자책 등록")
    if submitted and title.strip() and description.strip():
        file_text = ""
        if uploaded_file is not None:
            with st.spinner("파일 파싱 중..."):
                file_text = parse_file(uploaded_file)
        else:
            file_text = description
        with st.spinner("전자책 자동 분석 중..."):
            auto_analysis = ask_gpt(f"다음 전자책 내용을 요약하고, 개선점 및 중요한 부분을 알려줘:\n{file_text}")
        new_ebook = EBook(
            title=title,
            description=description,
            purchase_price=purchase_price,
            rental_price=rental_price,
            auto_analysis=auto_analysis,
            file_text=file_text
        )
        ebooks = st.session_state["ebooks"]
        ebooks.append(new_ebook)
        save_ebooks(ebooks)
        st.success("전자책 등록 및 자동 분석 완료!")

        # 블록체인에 전자책 정보 기록
        block_data = {
            "title": new_ebook.title,
            "description": new_ebook.description,
            "auto_analysis": new_ebook.auto_analysis,
            "purchase_price": new_ebook.purchase_price,
            "rental_price": new_ebook.rental_price
        }
        with st.spinner("전자책을 블록체인에 기록 중..."):
            idea_blockchain.add_block(block_data)
        st.info("전자책이 블록체인에 기록되었습니다.")

    st.write("---")
    st.write("### 등록된 전자책 목록")
    ebooks = st.session_state["ebooks"]
    if not ebooks:
        st.write("등록된 전자책이 없습니다.")
        return

    for idx, ebook in enumerate(ebooks):
        with st.expander(f"{idx+1}. {ebook.title}", expanded=False):
            st.write(f"**설명**: {ebook.description}")
            if ebook.auto_analysis:
                st.markdown("**전자책 자동 분석 결과:**")
                st.write(ebook.auto_analysis)
            st.markdown(f"**구매 가격**: {ebook.purchase_price}원  |  **대여 가격**: {ebook.rental_price}원")
            
            # 구매/대여 또는 멤버십 이용 기능
            if st.session_state["user_profile"]["membership"]:
                if st.button("멤버십으로 무료 이용", key=f"read_{idx}"):
                    st.write("### 전자책 내용")
                    st.write(ebook.file_text)
            else:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("구매하기", key=f"buy_{idx}"):
                        ebook.purchase_count += 1
                        save_ebooks(ebooks)
                        st.success("전자책 구매 완료!")
                with col2:
                    if st.button("대여하기", key=f"rent_{idx}"):
                        ebook.rental_count += 1
                        save_ebooks(ebooks)
                        st.success("전자책 대여 완료!")
            st.write(f"**구매 횟수**: {ebook.purchase_count}  |  **대여 횟수**: {ebook.rental_count}")

            st.write("### 리뷰")
            if not ebook.comments:
                st.write("리뷰가 없습니다.")
            else:
                for comment in ebook.comments:
                    st.write(f"- {comment}")
            new_comment = st.text_input("리뷰 작성", key=f"comment_{idx}")
            if st.button("리뷰 등록", key=f"submit_comment_{idx}"):
                if new_comment.strip():
                    ebook.comments.append(new_comment.strip())
                    save_ebooks(ebooks)
                    st.success("리뷰가 등록되었습니다.")
                else:
                    st.warning("리뷰 내용을 입력하세요.")

            if st.button("리뷰 분석", key=f"analyze_review_{idx}"):
                if ebook.comments:
                    reviews_text = "\n".join(ebook.comments)
                    with st.spinner("리뷰 분석 중..."):
                        analysis = ask_gpt(f"다음 전자책 리뷰를 분석하고, 개선점 및 중요한 피드백을 제공해줘:\n{reviews_text}")
                    ebook.review_analysis = analysis
                    save_ebooks(ebooks)
                    st.write("### 리뷰 분석 결과")
                    st.write(analysis)
                else:
                    st.warning("분석할 리뷰가 없습니다.")

            if st.button("전자책 삭제", key=f"delete_{idx}"):
                ebooks.pop(idx)
                st.session_state["ebooks"] = ebooks
                save_ebooks(ebooks)
                st.success("전자책이 삭제되었습니다.")

if __name__ == "__main__":
    main()
