import os
import nltk

nltk_data_dir = "/tmp/nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)
os.environ["NLTK_DATA"] = nltk_data_dir
nltk.data.path.append(nltk_data_dir)

nltk.download("stopwords", download_dir=nltk_data_dir)

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

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_dir)

korean_stopwords = [
    '이', '그', '저', '것', '수', '등', '들', '및', '더', '로', '를', '에',
    '의', '은', '는', '가', '와', '과', '하다', '있다', '되다', '이다',
    '으로', '에서', '까지', '부터', '만', '그리고', '하지만', '그러나'
]
english_stopwords = set(stopwords.words('english'))
korean_stopwords_set = set(korean_stopwords)
final_stopwords = english_stopwords.union(korean_stopwords_set)

st.set_page_config(page_title="studyhelper", layout="centered")

dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    openai_api_key = st.sidebar.text_input("OpenAI API 키를 입력하세요.", type="password")
    if not openai_api_key:
        st.error("OpenAI API 키가 설정되지 않았습니다.")
        st.stop()

openai.api_key = openai_api_key

# !!! 아래 3줄을 제거 or 주석 처리 !!!
# openai.api_base = "https://api.openai.com/v1"
# openai.api_type = None
# openai.api_version = None

try:
    st.write(f"OpenAI 라이브러리 버전: {openai.__version__}")
except:
    pass

def ask_gpt(prompt_text, model_name="gpt-4", temperature=0.0):
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt_text}
        ],
        temperature=temperature
    )
    return response.choices[0].message["content"].strip()

# 이하 동일 ...
