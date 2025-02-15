import os
import json
import openai
import streamlit as st
from dotenv import load_dotenv
import time
import hashlib
import base64

# -------------------------
# 환경 변수 로드 (OpenAI API)
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

# 전역 블록체인 인스턴스 (예: NFT 등록 내역 등)
idea_blockchain = Blockchain(difficulty=2)

# -------------------------
# AI 콘텐츠 모델 (image_url 필드 추가)
# -------------------------
class AIContent:
    def __init__(self, title, description, price, creator, file_text="", purchase_count=0, image_url=None):
        self.title = title
        self.description = description
        self.price = price
        self.creator = creator
        self.file_text = file_text
        self.purchase_count = purchase_count
        self.image_url = image_url  # 이미지 URL 추가

# -------------------------
# 전자책(JSON) 저장 (여기서는 AI 콘텐츠 저장)
# -------------------------
CONTENT_FILE = "ai_contents.json"

def load_contents():
    """
    기존 JSON 파일에 image_url 키가 없을 경우 None으로 설정하여 오류를 방지합니다.
    """
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
        # 기존 JSON에 image_url 키가 없으면 None을 반환
        image_url = item.get("image_url", None)
        content = AIContent(
            title=item["title"],
            description=item["description"],
            price=item["price"],
            creator=item["creator"],
            file_text=item.get("file_text", ""),
            purchase_count=item.get("purchase_count", 0),
            image_url=image_url
        )
        contents.append(content)
    return contents

def save_contents(contents):
    data = []
    for c in contents:
        data.append({
            "title": c.title,
            "description": c.description,
            "price": c.price,
            "creator": c.creator,
            "file_text": c.file_text,
            "purchase_count": c.purchase_count,
            "image_url": c.image_url
        })
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
# 이미지 생성 API 호출 (DALL·E)
# -------------------------
def generate_image(prompt):
    if not openai.api_key:
        st.error("⚠️ OpenAI API 키가 설정되지 않았습니다!")
        return None
    try:
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="512x512"  # 원하는 이미지 크기
        )
        image_url = response['data'][0]['url']
        st.write("✅ 이미지 생성 성공!")
        return image_url
    except Exception as e:
        st.error(f"⚠️ 이미지 생성 오류: {e}")
        return None

# -------------------------
# Web3 결제 시스템 (모의: 가상화폐 결제)
# -------------------------
def process_crypto_payment(amount):
    time.sleep(1)  # 결제 처리 모의 지연
    return True, f"✅ 결제 성공: {amount} 코인 전송 완료!"

# -------------------------
# 비트코인 잔액 조회 (모의 API)
# -------------------------
def fetch_bitcoin_balance():
    time.sleep(1)  # 조회 처리 모의 지연
    return "2.5 BTC"

# -------------------------
# 세션 초기화
# -------------------------
if "contents" not in st.session_state:
    st.session_state["contents"] = load_contents()

if "nfts" not in st.session_state:
    st.session_state["nfts"] = "[]"  # NFT 데이터를 JSON 문자열로 저장

if "user_profile" not in st.session_state:
    st.session_state["user_profile"] = {
        "username": "익명사용자",
        "experience": "개발, 스타트업 참여 경험 있음",
        "preferences": "핀테크, AI, 블록체인",
        "membership": False
    }

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
# 1) AI 콘텐츠 생성 & 업로드
# -------------------------
def create_ai_content():
    st.subheader("🧠 AI 콘텐츠 생성")
    title = st.text_input("📌 콘텐츠 제목")
    description = st.text_area("📄 설명")
    price = st.number_input("💰 가격 (가상화폐)", min_value=1, value=10)
    creator = st.text_input("✍️ 크리에이터 이름", "익명")

    # 이미지 업로드 또는 DALL·E 프롬프트
    st.markdown("**이미지 등록 방법**")
    col1, col2 = st.columns(2)
    with col1:
        uploaded_image = st.file_uploader("직접 업로드 (png, jpg, jpeg)", type=["png", "jpg", "jpeg"])
    with col2:
        image_prompt = st.text_input("DALL·E 프롬프트 (미업로드 시 자동생성)", value="귀여운 토끼 사진")

    if st.button("🎨 AI 콘텐츠 생성"):
        if not description:
            st.error("⚠️ 설명을 입력해주세요.")
            return

        with st.spinner("AI가 텍스트 콘텐츠 생성 중..."):
            file_text = generate_ai_content(description)

        # 이미지 처리
        image_url = None
        if uploaded_image is not None:
            # 예시: 이미지를 base64로 인코딩(임시). 
            # 실제로는 S3 등 클라우드에 업로드 후 해당 URL을 받는 방식을 권장.
            file_contents = uploaded_image.read()
            base64_img = base64.b64encode(file_contents).decode("utf-8")
            # data URI 스키마로 표시 (Streamlit의 st.image에서 인식 가능)
            image_url = f"data:image/png;base64,{base64_img}"
            st.success("이미지 업로드 완료!")
        else:
            # DALL·E로 자동 생성
            with st.spinner("DALL·E가 이미지를 생성 중..."):
                created_url = generate_image(image_prompt)
                if created_url:
                    image_url = created_url

        # AI 콘텐츠 객체 생성
        new_content = AIContent(
            title=title, 
            description=description, 
            price=price, 
            creator=creator, 
            file_text=file_text,
            image_url=image_url
        )

        # 세션 및 파일에 저장
        contents = st.session_state["contents"]
        contents.append(new_content)
        save_contents(contents)

        st.success("✅ AI 콘텐츠 생성 완료!")
        st.balloons()

# -------------------------
# 2) Web3 결제 & 마켓플레이스
# -------------------------
def content_marketplace():
    st.subheader("🛒 AI 콘텐츠 마켓플레이스")
    contents = st.session_state["contents"]
    if not contents:
        st.write("🚨 등록된 콘텐츠가 없습니다.")
        return

    for idx, content in enumerate(contents):
        with st.expander(f"{content.title}"):
            st.write(f"📝 설명: {content.description}")
            st.write(f"💰 가격: {content.price} 코인")
            st.write(f"🎨 크리에이터: {content.creator}")
            
            # 이미지 표시
            if content.image_url:
                st.image(content.image_url, use_column_width=True)

            # 결제 및 구매 버튼
            if st.button("💳 결제 및 구매", key=f"buy_{idx}"):
                success, message = process_crypto_payment(content.price)
                if success:
                    content.purchase_count += 1
                    save_contents(contents)
                    st.success(message)
                else:
                    st.error("⚠️ 결제 실패!")

# -------------------------
# 3) NFT 콘텐츠 거래 (NFT 등록 및 잔액 확인)
# -------------------------
def nft_marketplace():
    st.subheader("🖼 NFT 마켓플레이스")
    # 비트코인 잔액 표시
    btc_balance = fetch_bitcoin_balance()
    st.info(f"현재 비트코인 잔액: {btc_balance}")
    st.write("🚀 AI 콘텐츠를 NFT로 등록하고 거래하세요!")
    
    st.markdown("### NFT 등록")
    with st.form(key="nft_form", clear_on_submit=True):
        nft_title = st.text_input("NFT 제목")
        nft_description = st.text_area("NFT 설명")
        nft_price = st.number_input("NFT 가격 (코인)", min_value=1, value=10)
        nft_image = st.file_uploader("NFT 이미지 업로드", type=["png", "jpg", "jpeg"])
        # 이미지 업로드가 없는 경우 이미지 생성 프롬프트 입력
        image_prompt = st.text_input("이미지 생성 프롬프트 (이미지 업로드 없을 경우)", value="창의적인 NFT 아트워크")
        submitted_nft = st.form_submit_button("NFT 등록")
    
    if submitted_nft:
        if nft_image is not None:
            with st.spinner("이미지 분석 중..."):
                time.sleep(2)
                analysis_result = "분석 결과: 이 이미지는 창의적이고 독창적입니다."
            # 실제 파일 저장 또는 임시 base64 인코딩 (예시)
            file_contents = nft_image.read()
            base64_img = base64.b64encode(file_contents).decode("utf-8")
            image_url = f"data:image/png;base64,{base64_img}"
            st.success("이미지 업로드 및 분석 완료!")
        else:
            with st.spinner("이미지 생성 중..."):
                created_url = generate_image(image_prompt)
            if created_url:
                image_url = created_url
                analysis_result = "이미지 자동 생성"
            else:
                st.error("이미지 생성에 실패했습니다.")
                return
        
        nft = {
            "id": int(time.time()),
            "title": nft_title,
            "description": nft_description + "\n" + analysis_result,
            "price": nft_price,
            "imageURL": image_url,
            "owner": st.session_state["user_profile"]["username"]
        }
        nfts = json.loads(st.session_state["nfts"])
        nfts.append(nft)
        st.session_state["nfts"] = json.dumps(nfts, ensure_ascii=False, indent=2)
        st.success("✅ NFT 등록 완료되었습니다!")
    
    st.markdown("### 등록된 NFT")
    nfts = json.loads(st.session_state["nfts"])
    if not nfts:
        st.write("등록된 NFT가 없습니다.")
    else:
        for nft in nfts:
            st.write(f"**제목:** {nft['title']}")
            st.write(f"**설명:** {nft['description']}")
            st.write(f"**가격:** {nft['price']} 코인")
            st.write(f"**소유자:** {nft['owner']}")
            if nft['imageURL']:
                st.image(nft['imageURL'], width=250)
            st.write("---")

# -------------------------
# 앱 실행
# -------------------------
if __name__ == "__main__":
    main()
