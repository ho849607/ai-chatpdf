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
# AI 콘텐츠 모델 (저작권/대여 관련 필드 추가)
# -------------------------
class AIContent:
    def __init__(self, title, description, price, creator, file_text="", purchase_count=0, image_url=None,
                 copyright_registered=False, copyright_cert="",
                 copyright_eligibility="",
                 copyright_lease_requested=False,
                 lease_conditions="",
                 lease_contract="",
                 lease_eligibility=""):
        self.title = title
        self.description = description
        self.price = price
        self.creator = creator
        self.file_text = file_text
        self.purchase_count = purchase_count
        self.image_url = image_url
        # 저작권 등록 정보
        self.copyright_registered = copyright_registered
        self.copyright_cert = copyright_cert
        self.copyright_eligibility = copyright_eligibility
        # 저작권 대여(라이선스) 관련 정보
        self.copyright_lease_requested = copyright_lease_requested
        self.lease_conditions = lease_conditions
        self.lease_contract = lease_contract
        self.lease_eligibility = lease_eligibility

# -------------------------
# 전자책(JSON) 저장 (여기서는 AI 콘텐츠 저장)
# -------------------------
CONTENT_FILE = "ai_contents.json"

def load_contents():
    """
    기존 JSON 파일에 image_url 및 저작권/대여 관련 키가 없을 경우 기본값으로 처리
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
        content = AIContent(
            title=item["title"],
            description=item["description"],
            price=item["price"],
            creator=item["creator"],
            file_text=item.get("file_text", ""),
            purchase_count=item.get("purchase_count", 0),
            image_url=item.get("image_url", None),
            copyright_registered=item.get("copyright_registered", False),
            copyright_cert=item.get("copyright_cert", ""),
            copyright_eligibility=item.get("copyright_eligibility", ""),
            copyright_lease_requested=item.get("copyright_lease_requested", False),
            lease_conditions=item.get("lease_conditions", ""),
            lease_contract=item.get("lease_contract", ""),
            lease_eligibility=item.get("lease_eligibility", "")
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
            "image_url": c.image_url,
            "copyright_registered": c.copyright_registered,
            "copyright_cert": c.copyright_cert,
            "copyright_eligibility": c.copyright_eligibility,
            "copyright_lease_requested": c.copyright_lease_requested,
            "lease_conditions": c.lease_conditions,
            "lease_contract": c.lease_contract,
            "lease_eligibility": c.lease_eligibility
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
# 저작권 등록 (모의 기능)
# -------------------------
def register_copyright(image_url):
    """
    이미지 URL을 이용해 (모의) 저작권 등록을 진행.
    등록 ID는 이미지 URL의 해시 일부를 사용하며, 항상 '저작권 인정 가능'으로 처리합니다.
    """
    if not image_url:
        return None, "이미지 없음"
    registration_id = f"COPY-{hashlib.sha256(image_url.encode()).hexdigest()[:10]}"
    eligibility = "저작권 인정 가능"
    return registration_id, eligibility

# -------------------------
# 저작권 대여(라이선스) 등록 (모의 기능)
# -------------------------
def register_copyright_lease(image_url, lease_conditions):
    """
    이미지 URL과 대여 조건을 기반으로 (모의) 저작권 대여 계약을 진행합니다.
    계약 ID는 이미지 URL과 대여 조건을 해시하여 생성하며, '대여 가능' 상태를 반환합니다.
    """
    if not image_url or not lease_conditions.strip():
        return None, "대여 조건 미설정"
    contract_id = f"LEASE-{hashlib.sha256((image_url + lease_conditions).encode()).hexdigest()[:10]}"
    eligibility = "대여 가능"
    return contract_id, eligibility

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

    # 저작권 등록 요청 옵션
    copyright_option = st.checkbox("저작권 등록 요청", value=False)
    # 저작권 대여(라이선스) 요청 옵션 및 조건 입력
    lease_option = st.checkbox("저작권 대여 서비스 요청", value=False)
    lease_conditions_input = ""
    if lease_option:
        lease_conditions_input = st.text_area("대여 조건 입력 (예: 대여 기간, 비용 등)")

    if st.button("🎨 AI 콘텐츠 생성"):
        if not description:
            st.error("⚠️ 설명을 입력해주세요.")
            return

        with st.spinner("AI가 텍스트 콘텐츠 생성 중..."):
            file_text = generate_ai_content(description)

        # 이미지 처리
        image_url = None
        if uploaded_image is not None:
            file_contents = uploaded_image.read()
            base64_img = base64.b64encode(file_contents).decode("utf-8")
            image_url = f"data:image/png;base64,{base64_img}"
            st.success("이미지 업로드 완료!")
        else:
            with st.spinner("DALL·E가 이미지를 생성 중..."):
                created_url = generate_image(image_prompt)
                if created_url:
                    image_url = created_url

        # 저작권 등록 처리 (옵션 선택 시)
        copyright_registered = False
        copyright_cert = ""
        copyright_eligibility = ""
        if copyright_option and image_url:
            copyright_cert, copyright_eligibility = register_copyright(image_url)
            copyright_registered = True

        # 저작권 대여(라이선스) 처리 (옵션 선택 시)
        lease_contract = ""
        lease_eligibility = ""
        if lease_option and image_url and lease_conditions_input.strip():
            lease_contract, lease_eligibility = register_copyright_lease(image_url, lease_conditions_input)

        # AI 콘텐츠 객체 생성
        new_content = AIContent(
            title=title, 
            description=description, 
            price=price, 
            creator=creator, 
            file_text=file_text,
            image_url=image_url,
            copyright_registered=copyright_registered,
            copyright_cert=copyright_cert,
            copyright_eligibility=copyright_eligibility,
            copyright_lease_requested=lease_option,
            lease_conditions=lease_conditions_input,
            lease_contract=lease_contract,
            lease_eligibility=lease_eligibility
        )

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
            
            if content.image_url:
                st.image(content.image_url, use_column_width=True)

            # 저작권 등록 정보 표시
            if content.image_url:
                if content.copyright_registered:
                    st.write(f"🔒 저작권 등록 완료: {content.copyright_cert}")
                    st.write(f"📌 상태: {content.copyright_eligibility}")
                else:
                    st.write("🆓 저작권 미등록")
            
            # 저작권 대여(라이선스) 정보 표시
            if content.copyright_lease_requested:
                if content.lease_contract:
                    st.write(f"💼 대여 계약 ID: {content.lease_contract}")
                    st.write(f"📌 대여 조건: {content.lease_eligibility}")
                    st.write(f"📝 대여 조건 상세: {content.lease_conditions}")
                else:
                    st.write("💼 대여 서비스 요청됨 (조건 미설정)")

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
    btc_balance = fetch_bitcoin_balance()
    st.info(f"현재 비트코인 잔액: {btc_balance}")
    st.write("🚀 AI 콘텐츠를 NFT로 등록하고 거래하세요!")
    
    st.markdown("### NFT 등록")
    with st.form(key="nft_form", clear_on_submit=True):
        nft_title = st.text_input("NFT 제목")
        nft_description = st.text_area("NFT 설명")
        nft_price = st.number_input("NFT 가격 (코인)", min_value=1, value=10)
        nft_image = st.file_uploader("NFT 이미지 업로드", type=["png", "jpg", "jpeg"])
        image_prompt = st.text_input("이미지 생성 프롬프트 (이미지 업로드 없을 경우)", value="창의적인 NFT 아트워크")
        # NFT 등록 시에도 저작권 등록 및 대여 서비스 옵션 제공
        nft_copyright_option = st.checkbox("저작권 등록 요청", value=False)
        nft_lease_option = st.checkbox("저작권 대여 서비스 요청", value=False)
        nft_lease_conditions = ""
        if nft_lease_option:
            nft_lease_conditions = st.text_area("대여 조건 입력 (예: 대여 기간, 비용 등)")
        submitted_nft = st.form_submit_button("NFT 등록")
    
    if submitted_nft:
        if nft_image is not None:
            with st.spinner("이미지 분석 중..."):
                time.sleep(2)
                analysis_result = "분석 결과: 이 이미지는 창의적이고 독창적입니다."
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
        
        # NFT 저작권 등록 처리 (옵션)
        nft_copyright_registered = False
        nft_copyright_cert = ""
        nft_copyright_eligibility = ""
        if nft_copyright_option and image_url:
            nft_copyright_cert, nft_copyright_eligibility = register_copyright(image_url)
            nft_copyright_registered = True

        # NFT 저작권 대여(라이선스) 처리 (옵션)
        nft_lease_contract = ""
        nft_lease_eligibility = ""
        if nft_lease_option and image_url and nft_lease_conditions.strip():
            nft_lease_contract, nft_lease_eligibility = register_copyright_lease(image_url, nft_lease_conditions)

        nft = {
            "id": int(time.time()),
            "title": nft_title,
            "description": nft_description + "\n" + analysis_result,
            "price": nft_price,
            "imageURL": image_url,
            "owner": st.session_state["user_profile"]["username"],
            "copyright_registered": nft_copyright_registered,
            "copyright_cert": nft_copyright_cert,
            "copyright_eligibility": nft_copyright_eligibility,
            "copyright_lease_requested": nft_lease_option,
            "lease_conditions": nft_lease_conditions,
            "lease_contract": nft_lease_contract,
            "lease_eligibility": nft_lease_eligibility
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
            if nft.get("copyright_registered"):
                st.write(f"🔒 저작권 등록 완료: {nft.get('copyright_cert')}")
                st.write(f"📌 상태: {nft.get('copyright_eligibility')}")
            else:
                st.write("🆓 저작권 미등록")
            if nft.get("copyright_lease_requested"):
                if nft.get("lease_contract"):
                    st.write(f"💼 대여 계약 ID: {nft.get('lease_contract')}")
                    st.write(f"📌 대여 조건: {nft.get('lease_eligibility')}")
                    st.write(f"📝 대여 조건 상세: {nft.get('lease_conditions')}")
                else:
                    st.write("💼 대여 서비스 요청됨 (조건 미설정)")
            st.write("---")

# -------------------------
# 앱 실행
# -------------------------
if __name__ == "__main__":
    main()
