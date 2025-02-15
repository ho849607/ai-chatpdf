import os
import json
import openai
import streamlit as st
from dotenv import load_dotenv
import time
import hashlib
import base64

# -------------------------
# 번역 문자열 (한국어/English)
# -------------------------
translations = {
    "lang_title": {"ko": "언어", "en": "Language"},
    "title": {"ko": "🚀 Sharehost: AI 콘텐츠 & Web3 결제", "en": "🚀 Sharehost: AI Content & Web3 Payment"},
    "menu_AI": {"ko": "AI 콘텐츠 생성", "en": "AI Content Creation"},
    "menu_Web3": {"ko": "Web3 결제 & 마켓플레이스", "en": "Web3 Payment & Marketplace"},
    "menu_NFT": {"ko": "NFT 콘텐츠 거래", "en": "NFT Content Trading (NFT)"},
    "create_ai_subheader": {"ko": "🧠 AI 콘텐츠 생성", "en": "🧠 AI Content Creation"},
    "content_title": {"ko": "📌 콘텐츠 제목", "en": "📌 Content Title"},
    "description": {"ko": "📄 설명", "en": "📄 Description"},
    "price_coin": {"ko": "💰 가격 (코인 단위)", "en": "💰 Price (in Coins)"},
    "creator": {"ko": "✍️ 크리에이터 이름", "en": "✍️ Creator Name"},
    "image_registration": {"ko": "**이미지 등록 방법**", "en": "**Image Registration Method**"},
    "upload_direct": {"ko": "직접 업로드 (png, jpg, jpeg)", "en": "Upload directly (png, jpg, jpeg)"},
    "dalle_prompt": {"ko": "DALL·E 프롬프트 (미업로드 시 자동생성)", "en": "DALL·E Prompt (Auto generate if not uploaded)"},
    "copyright_request": {"ko": "저작권 등록 요청", "en": "Request Copyright Registration"},
    "lease_request": {"ko": "저작권 대여 서비스 요청", "en": "Request Copyright Lease Service"},
    "lease_conditions": {"ko": "대여 조건 입력 (예: 대여 기간, 비용 등)", "en": "Enter lease conditions (e.g., duration, cost, etc.)"},
    "generate_ai_button": {"ko": "🎨 AI 콘텐츠 생성", "en": "🎨 Generate AI Content"},
    "enter_description": {"ko": "설명을 입력해주세요.", "en": "Please enter a description."},
    "generating_text": {"ko": "AI가 텍스트 콘텐츠 생성 중...", "en": "Generating text content with AI..."},
    "upload_success": {"ko": "이미지 업로드 완료!", "en": "Image uploaded successfully!"},
    "generating_image": {"ko": "DALL·E가 이미지를 생성 중...", "en": "Generating image using DALL·E..."},
    "copyright_registered_text": {"ko": "저작권 등록 완료", "en": "Copyright Registered"},
    "no_copyright": {"ko": "저작권 미등록", "en": "No Copyright Registration"},
    "lease_contract_id": {"ko": "대여 계약 ID", "en": "Lease Contract ID"},
    "lease_conditions_label": {"ko": "대여 조건", "en": "Lease Conditions"},
    "lease_conditions_detail": {"ko": "대여 조건 상세", "en": "Detailed Lease Conditions"},
    "ai_content_success": {"ko": "✅ AI 콘텐츠 생성 완료!", "en": "✅ AI Content Generation Complete!"},
    "marketplace_subheader": {"ko": "🛒 AI 콘텐츠 마켓플레이스", "en": "🛒 AI Content Marketplace"},
    "no_content": {"ko": "등록된 콘텐츠가 없습니다.", "en": "No content registered."},
    "label_description": {"ko": "📝 설명:", "en": "📝 Description:"},
    "label_price": {"ko": "💰 가격:", "en": "💰 Price:"},
    "label_creator": {"ko": "🎨 크리에이터:", "en": "🎨 Creator:"},
    "purchase_button": {"ko": "💳 결제 및 구매", "en": "💳 Purchase"},
    "payment_success": {"ko": "결제 성공", "en": "Payment Successful"},
    "payment_failure": {"ko": "결제 실패", "en": "Payment Failed"},
    "nft_marketplace_subheader": {"ko": "🖼 NFT 마켓플레이스", "en": "🖼 NFT Marketplace"},
    "bitcoin_balance": {"ko": "현재 비트코인 잔액:", "en": "Current Bitcoin Balance:"},
    "nft_register_instruction": {"ko": "🚀 AI 콘텐츠를 NFT로 등록하고 거래하세요!", "en": "Register and trade your AI content as NFTs!"},
    "nft_registration": {"ko": "NFT 등록", "en": "Register NFT"},
    "nft_title": {"ko": "NFT 제목", "en": "NFT Title"},
    "nft_description": {"ko": "NFT 설명", "en": "NFT Description"},
    "nft_price": {"ko": "NFT 가격 (코인 단위)", "en": "NFT Price (in Coins)"},
    "nft_upload_image": {"ko": "NFT 이미지 업로드", "en": "Upload NFT Image"},
    "nft_dalle_prompt": {"ko": "이미지 생성 프롬프트 (미업로드 시 자동생성)", "en": "Image Generation Prompt (Auto generate if not uploaded)"},
    "nft_register_success": {"ko": "NFT 등록 완료되었습니다!", "en": "NFT Registered Successfully!"},
    "no_nft": {"ko": "등록된 NFT가 없습니다.", "en": "No NFTs registered."},
    "coin": {"ko": "코인", "en": "Coin"},
    "krw": {"ko": "원", "en": "KRW"},
    "usd": {"ko": "USD", "en": "USD"},
    "lease_requested_text": {"ko": "대여 서비스 요청됨 (조건 미설정)", "en": "Lease service requested (conditions not set)"},
}

def tr(key, lang):
    return translations.get(key, {}).get(lang, key)

# -------------------------
# 환율 상수 (예시: 1 코인 = 1,000원, 1 코인 = 1.3USD)
# -------------------------
KRW_RATE = 1000  # 1 코인당 1,000원
USD_RATE = 1.3   # 1 코인당 1.3 달러

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
# 사이드바 언어 선택
# -------------------------
language_choice = st.sidebar.radio(tr("lang_title", "en"), ["한국어", "English"])
lang = "ko" if language_choice == "한국어" else "en"

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
        st.write(f"🔗 {tr('copyright_registered_text', lang)}: {self.hash}")

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
        return f"⚠️ 오류: OpenAI API 키 필요"
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
        st.error(f"⚠️ OpenAI API 키가 설정되지 않았습니다!")
        return None
    try:
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="512x512"
        )
        image_url = response['data'][0]['url']
        st.write(f"✅ {tr('upload_success', lang)}")
        return image_url
    except Exception as e:
        st.error(f"⚠️ 이미지 생성 오류: {e}")
        return None

# -------------------------
# 저작권 등록 (모의 기능)
# -------------------------
def register_copyright(image_url):
    if not image_url:
        return None, "이미지 없음"
    registration_id = f"COPY-{hashlib.sha256(image_url.encode()).hexdigest()[:10]}"
    eligibility = tr("copyright_registered_text", lang)
    return registration_id, eligibility

# -------------------------
# 저작권 대여(라이선스) 등록 (모의 기능)
# -------------------------
def register_copyright_lease(image_url, lease_conditions):
    if not image_url or not lease_conditions.strip():
        return None, "대여 조건 미설정"
    contract_id = f"LEASE-{hashlib.sha256((image_url + lease_conditions).encode()).hexdigest()[:10]}"
    eligibility = "대여 가능" if lang == "ko" else "Lease Available"
    return contract_id, eligibility

# -------------------------
# Web3 결제 시스템 (모의: 가상화폐 결제)
# -------------------------
def process_crypto_payment(amount):
    time.sleep(1)
    return True, f"✅ {tr('payment_success', lang)}: {amount} {tr('coin', lang)} 전송 완료!"

# -------------------------
# 비트코인 잔액 조회 (모의 API)
# -------------------------
def fetch_bitcoin_balance():
    time.sleep(1)
    return "2.5 BTC"

# -------------------------
# 세션 초기화
# -------------------------
if "contents" not in st.session_state:
    st.session_state["contents"] = load_contents()

if "nfts" not in st.session_state:
    st.session_state["nfts"] = "[]"

if "user_profile" not in st.session_state:
    st.session_state["user_profile"] = {
        "username": "익명사용자" if lang == "ko" else "Anonymous",
        "experience": "개발, 스타트업 참여 경험 있음" if lang == "ko" else "Experience in development and startups",
        "preferences": "핀테크, AI, 블록체인" if lang == "ko" else "Fintech, AI, Blockchain",
        "membership": False
    }

# -------------------------
# 메인 Streamlit 앱
# -------------------------
def main():
    st.title(tr("title", lang))
    menu = st.sidebar.radio("", [tr("menu_AI", lang), tr("menu_Web3", lang), tr("menu_NFT", lang)])
    if menu == tr("menu_AI", lang):
        create_ai_content()
    elif menu == tr("menu_Web3", lang):
        content_marketplace()
    else:
        nft_marketplace()

# -------------------------
# 1) AI 콘텐츠 생성 & 업로드
# -------------------------
def create_ai_content():
    st.subheader(tr("create_ai_subheader", lang))
    title = st.text_input(tr("content_title", lang))
    description = st.text_area(tr("description", lang))
    price = st.number_input(tr("price_coin", lang), min_value=1, value=10)
    creator = st.text_input(tr("creator", lang), "익명" if lang == "ko" else "Anonymous")

    st.markdown(tr("image_registration", lang))
    col1, col2 = st.columns(2)
    with col1:
        uploaded_image = st.file_uploader(tr("upload_direct", lang), type=["png", "jpg", "jpeg"])
    with col2:
        image_prompt = st.text_input(tr("dalle_prompt", lang), value="귀여운 토끼 사진" if lang == "ko" else "Cute rabbit photo")

    copyright_option = st.checkbox(tr("copyright_request", lang), value=False)
    lease_option = st.checkbox(tr("lease_request", lang), value=False)
    lease_conditions_input = ""
    if lease_option:
        lease_conditions_input = st.text_area(tr("lease_conditions", lang))
        
    if st.button(tr("generate_ai_button", lang)):
        if not description:
            st.error(tr("enter_description", lang))
            return

        with st.spinner(tr("generating_text", lang)):
            file_text = generate_ai_content(description)

        image_url = None
        if uploaded_image is not None:
            file_contents = uploaded_image.read()
            base64_img = base64.b64encode(file_contents).decode("utf-8")
            image_url = f"data:image/png;base64,{base64_img}"
            st.success(tr("upload_success", lang))
        else:
            with st.spinner(tr("generating_image", lang)):
                created_url = generate_image(image_prompt)
                if created_url:
                    image_url = created_url

        cr_registered = False
        cr_cert = ""
        cr_eligibility = ""
        if copyright_option and image_url:
            cr_cert, cr_eligibility = register_copyright(image_url)
            cr_registered = True

        lease_contract = ""
        lease_eligibility = ""
        if lease_option and image_url and lease_conditions_input.strip():
            lease_contract, lease_eligibility = register_copyright_lease(image_url, lease_conditions_input)

        new_content = AIContent(
            title=title, 
            description=description, 
            price=price, 
            creator=creator, 
            file_text=file_text,
            image_url=image_url,
            copyright_registered=cr_registered,
            copyright_cert=cr_cert,
            copyright_eligibility=cr_eligibility,
            copyright_lease_requested=lease_option,
            lease_conditions=lease_conditions_input,
            lease_contract=lease_contract,
            lease_eligibility=lease_eligibility
        )

        contents = st.session_state["contents"]
        contents.append(new_content)
        save_contents(contents)

        st.success(tr("ai_content_success", lang))
        st.balloons()

# -------------------------
# 2) Web3 결제 & 마켓플레이스
# -------------------------
def content_marketplace():
    st.subheader(tr("marketplace_subheader", lang))
    contents = st.session_state["contents"]
    if not contents:
        st.write(tr("no_content", lang))
        return

    for idx, content in enumerate(contents):
        with st.expander(f"{content.title}"):
            st.write(f"{tr('label_description', lang)} {content.description}")
            price_text = (f"{content.price} {tr('coin', lang)} / "
                          f"{content.price * KRW_RATE} {tr('krw', lang)} / "
                          f"{content.price * USD_RATE:.2f} {tr('usd', lang)}")
            st.write(f"{tr('label_price', lang)} {price_text}")
            st.write(f"{tr('label_creator', lang)} {content.creator}")
            
            if content.image_url:
                st.image(content.image_url, use_column_width=True)

            if content.image_url:
                if content.copyright_registered:
                    st.write(f"🔒 {tr('copyright_registered_text', lang)}: {content.copyright_cert}")
                    st.write(f"📌 {content.copyright_eligibility}")
                else:
                    st.write(f"🆓 {tr('no_copyright', lang)}")
            
            if content.copyright_lease_requested:
                if content.lease_contract:
                    st.write(f"💼 {tr('lease_contract_id', lang)}: {content.lease_contract}")
                    st.write(f"📌 {tr('lease_conditions_label', lang)}: {content.lease_eligibility}")
                    st.write(f"📝 {tr('lease_conditions_detail', lang)}: {content.lease_conditions}")
                else:
                    st.write(tr("lease_requested_text", lang))

            if st.button(tr("purchase_button", lang), key=f"buy_{idx}"):
                success, message = process_crypto_payment(content.price)
                if success:
                    content.purchase_count += 1
                    save_contents(contents)
                    st.success(message)
                else:
                    st.error(tr("payment_failure", lang))

# -------------------------
# 3) NFT 콘텐츠 거래 (NFT 등록 및 잔액 확인)
# -------------------------
def nft_marketplace():
    st.subheader(tr("nft_marketplace_subheader", lang))
    btc_balance = fetch_bitcoin_balance()
    st.info(f"{tr('bitcoin_balance', lang)} {btc_balance}")
    st.write(tr("nft_register_instruction", lang))
    
    st.markdown(f"### {tr('nft_registration', lang)}")
    with st.form(key="nft_form", clear_on_submit=True):
        nft_title = st.text_input(tr("nft_title", lang))
        nft_description = st.text_area(tr("nft_description", lang))
        nft_price = st.number_input(tr("nft_price", lang), min_value=1, value=10)
        nft_image = st.file_uploader(tr("nft_upload_image", lang), type=["png", "jpg", "jpeg"])
        image_prompt = st.text_input(tr("nft_dalle_prompt", lang), value="창의적인 NFT 아트워크" if lang == "ko" else "Creative NFT Artwork")
        nft_copyright_option = st.checkbox(tr("copyright_request", lang), value=False)
        nft_lease_option = st.checkbox(tr("lease_request", lang), value=False)
        nft_lease_conditions = ""
        if nft_lease_option:
            nft_lease_conditions = st.text_area(tr("lease_conditions", lang))
        submitted_nft = st.form_submit_button(tr("nft_registration", lang))
    
    if submitted_nft:
        if nft_image is not None:
            with st.spinner("이미지 분석 중..."):
                time.sleep(2)
                analysis_result = "분석 결과: 이 이미지는 창의적이고 독창적입니다." if lang == "ko" else "Analysis: This image is creative and unique."
            file_contents = nft_image.read()
            base64_img = base64.b64encode(file_contents).decode("utf-8")
            image_url = f"data:image/png;base64,{base64_img}"
            st.success("이미지 업로드 및 분석 완료!" if lang == "ko" else "Image upload and analysis complete!")
        else:
            with st.spinner(tr("generating_image", lang)):
                created_url = generate_image(image_prompt)
            if created_url:
                image_url = created_url
                analysis_result = "이미지 자동 생성" if lang == "ko" else "Image auto-generated"
            else:
                st.error("이미지 생성에 실패했습니다." if lang == "ko" else "Image generation failed.")
                return
        
        nft_cr_registered = False
        nft_cr_cert = ""
        nft_cr_eligibility = ""
        if nft_copyright_option and image_url:
            nft_cr_cert, nft_cr_eligibility = register_copyright(image_url)
            nft_cr_registered = True

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
            "copyright_registered": nft_cr_registered,
            "copyright_cert": nft_cr_cert,
            "copyright_eligibility": nft_cr_eligibility,
            "copyright_lease_requested": nft_lease_option,
            "lease_conditions": nft_lease_conditions,
            "lease_contract": nft_lease_contract,
            "lease_eligibility": nft_lease_eligibility
        }
        nfts = json.loads(st.session_state["nfts"])
        nfts.append(nft)
        st.session_state["nfts"] = json.dumps(nfts, ensure_ascii=False, indent=2)
        st.success(tr("nft_register_success", lang))
    
    st.markdown(f"### {tr('no_nft', lang) if json.loads(st.session_state['nfts'])==[] else ''}")
    nfts = json.loads(st.session_state["nfts"])
    if not nfts:
        st.write(tr("no_nft", lang))
    else:
        for nft in nfts:
            st.write(f"**{tr('nft_title', lang)}:** {nft['title']}")
            st.write(f"**{tr('nft_description', lang)}:** {nft['description']}")
            nft_price_text = (f"{nft['price']} {tr('coin', lang)} / "
                              f"{nft['price'] * KRW_RATE} {tr('krw', lang)} / "
                              f"{nft['price'] * USD_RATE:.2f} {tr('usd', lang)}")
            st.write(f"**{tr('label_price', lang)}** {nft_price_text}")
            st.write(f"**{tr('label_creator', lang)}** {nft['owner']}")
            if nft['imageURL']:
                st.image(nft['imageURL'], width=250)
            if nft.get("copyright_registered"):
                st.write(f"🔒 {tr('copyright_registered_text', lang)}: {nft.get('copyright_cert')}")
                st.write(f"📌 {nft.get('copyright_eligibility')}")
            else:
                st.write(f"🆓 {tr('no_copyright', lang)}")
            if nft.get("copyright_lease_requested"):
                if nft.get("lease_contract"):
                    st.write(f"💼 {tr('lease_contract_id', lang)}: {nft.get('lease_contract')}")
                    st.write(f"📌 {tr('lease_conditions_label', lang)}: {nft.get('lease_eligibility')}")
                    st.write(f"📝 {tr('lease_conditions_detail', lang)}: {nft.get('lease_conditions')}")
                else:
                    st.write(tr("lease_requested_text", lang))
            st.write("---")

# -------------------------
# 앱 실행
# -------------------------
if __name__ == "__main__":
    main()
