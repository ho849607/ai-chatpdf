import os
import json
import openai
import streamlit as st
from dotenv import load_dotenv
import time
import hashlib

# -------------------------
# 환경 변수 로드 (OpenAI API + 블록체인 연결)
# -------------------------
load_dotenv('.env')
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    st.warning("⚠️ OPENAI_API_KEY가 필요합니다.")
else:
    openai.api_key = openai_api_key

# -------------------------
# 블록체인 구현 (NFT & AI 콘텐츠 인증)
# -------------------------
class Block:
    def __init__(self, index, timestamp, data, previous_hash, nonce=0):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce
        }, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def mine_block(self, difficulty):
        target = "0" * difficulty
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()
        st.write(f"🔗 블록 생성 완료: {self.hash}")

class Blockchain:
    def __init__(self, difficulty=2):
        self.chain = [self.create_genesis_block()]
        self.difficulty = difficulty

    def create_genesis_block(self):
        return Block(0, time.time(), "Genesis Block", "0")

    def get_latest_block(self):
        return self.chain[-1]

    def add_block(self, data):
        new_index = len(self.chain)
        new_block = Block(new_index, time.time(), data, self.get_latest_block().hash)
        new_block.mine_block(self.difficulty)
        self.chain.append(new_block)

# -------------------------
# AI 콘텐츠 생성 & Web3 결제 시스템
# -------------------------
class AIContent:
    def __init__(self, title, description, price, creator, file_text="", purchase_count=0):
        self.title = title
        self.description = description
        self.price = price
        self.creator = creator
        self.file_text = file_text
        self.purchase_count = purchase_count

# -------------------------
# 전자책 JSON 저장
# -------------------------
CONTENT_FILE = "ai_contents.json"

def load_contents():
    if not os.path.exists(CONTENT_FILE):
        return []
    try:
        with open(CONTENT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        st.error(f"⚠️ 데이터 로드 오류: {e}")
        return []
    
    contents = []
    for item in data:
        contents.append(AIContent(**item))
    return contents

def save_contents(contents):
    data = [vars(content) for content in contents]
    try:
        with open(CONTENT_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"⚠️ 데이터 저장 오류: {e}")

# -------------------------
# GPT API 호출 (AI 콘텐츠 자동 생성)
# -------------------------
def generate_ai_content(prompt):
    if not openai.api_key:
        return "⚠️ 오류: OpenAI API 키 필요"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"⚠️ 오류 발생: {e}"

# -------------------------
# Web3 결제 시스템 (가상화폐 트랜잭션 시뮬레이션)
# -------------------------
def process_crypto_payment(amount):
    time.sleep(1)
    return True, f"✅ 결제 성공: {amount} 코인 전송 완료!"

# -------------------------
# 메인 Streamlit 앱
# -------------------------
def main():
    st.title("🚀 Sharehost: AI 콘텐츠 & Web3 결제")

    menu = st.sidebar.radio("메뉴", ["AI 콘텐츠 생성", "Web3 결제 & 마켓플레이스", "NFT 콘텐츠 거래"])

    if menu == "AI 콘텐츠 생성":
        create_ai_content()
    elif menu == "Web3 결제 & 마켓플레이스":
        content_marketplace()
    else:
        nft_marketplace()

# -------------------------
# AI 콘텐츠 생성 & 업로드
# -------------------------
def create_ai_content():
    st.subheader("🧠 AI 콘텐츠 생성")
    title = st.text_input("📌 콘텐츠 제목")
    description = st.text_area("📄 설명")
    price = st.number_input("💰 가격 (가상화폐)", min_value=1, value=10)
    creator = st.text_input("✍️ 크리에이터 이름", "익명")

    if st.button("🎨 AI 콘텐츠 생성"):
        with st.spinner("AI가 콘텐츠 생성 중..."):
            file_text = generate_ai_content(description)
            new_content = AIContent(title, description, price, creator, file_text)
            contents = load_contents()
            contents.append(new_content)
            save_contents(contents)
            st.success("✅ AI 콘텐츠 생성 완료!")

# -------------------------
# Web3 결제 & 마켓플레이스
# -------------------------
def content_marketplace():
    st.subheader("🛒 AI 콘텐츠 마켓플레이스")
    contents = load_contents()
    
    if not contents:
        st.write("🚨 등록된 콘텐츠가 없습니다.")
        return

    for idx, content in enumerate(contents):
        with st.expander(f"{idx+1}. {content.title}"):
            st.write(f"📝 설명: {content.description}")
            st.write(f"💰 가격: {content.price} 코인")
            st.write(f"🎨 크리에이터: {content.creator}")
            if st.button("💳 결제 및 구매", key=f"buy_{idx}"):
                success, message = process_crypto_payment(content.price)
                if success:
                    content.purchase_count += 1
                    save_contents(contents)
                    st.success(message)

# -------------------------
# NFT 콘텐츠 거래 (Web3 결제 시스템)
# -------------------------
def nft_marketplace():
    st.subheader("🖼 NFT 마켓플레이스")
    st.write("🚀 AI 콘텐츠를 NFT로 등록하고 거래하세요!")

    if st.button("🎨 NFT 등록하기"):
        st.success("✅ NFT 등록 완료!")

# -------------------------
# 실행
# -------------------------
if __name__ == "__main__":
    main()
