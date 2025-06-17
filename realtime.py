#python -m streamlit run realtime.py   

import os
import streamlit as st
import openai
from transformers import pipeline
from google.cloud import texttospeech, speech
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder
from konlpy.tag import Okt

# ─────────────────────────────────────────────
# 페이지 설정 및 사이드바
st.set_page_config(page_title="✨ 음성 클렌징 & 합성 데모", layout="centered")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2920/2920277.png", width=90)
    st.markdown("## 서비스 안내")
    st.info(
        "- **욕설/혐오 표현** 자동 탐지 및 순화\n"
        "- **TTS 합성** 및 다운로드 제공\n"
        "- 마이크로 직접 녹음 또는 음성 파일 업로드"
    )
    st.markdown("---")
    st.caption("ⓒ 2025 Soft Hate Speech Classifier Demo")

# ─────────────────────────────────────────────
# API 인증 및 클라이언트 설정
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "textmining-461305-f0cddebc86fe.json"

# ─────────────────────────────────────────────
# 리소스 로딩
@st.cache_resource
def load_speech_client():
    return speech.SpeechClient()

@st.cache_resource
def load_bad_words(filepath: str):
    if not os.path.exists(filepath): return set()
    with open(filepath, "r", encoding="utf-8") as f:
        return set(w.strip().lower() for w in f if w.strip())

@st.cache_resource
def load_hate_pipeline(model_path: str):
    return pipeline("text-classification", model=model_path, tokenizer=model_path)

# ─────────────────────────────────────────────
okt = Okt()

def contains_bad_word_loose(text: str, bad_words: set) -> bool:
    words = text.split()
    for word in words:
        if word.lower() in bad_words:
            return True
    tokens = okt.pos(text, norm=True, stem=True)
    for idx, (tok, tag) in enumerate(tokens):
        t = tok.lower()
        for bw in bad_words:
            if not bw:
                continue
            if t == bw:
                return True
            if t == '개':
                if idx + 1 < len(tokens) and tokens[idx + 1][0].lower() in bad_words:
                    return True
                if idx - 1 >= 0 and tokens[idx - 1][0].lower() in bad_words:
                    return True
                continue
            if t == '년':
                if idx - 1 >= 0 and tokens[idx - 1][1] == 'Number':
                    continue
                return True
            if t == '새끼':
                if idx + 1 < len(tokens) and tokens[idx + 1][1] == 'Noun' and tokens[idx + 1][0].lower() not in bad_words:
                    continue
                return True
            if t == '자식':
                if idx - 1 >= 0 and (tokens[idx - 1][0].lower() in bad_words or tokens[idx - 1][0].lower() == '개'):
                    return True
                if idx + 1 < len(tokens) and tokens[idx + 1][0].lower() in bad_words:
                    return True
                continue
    return False

def sanitize_text(text: str) -> str:
    prompt = f"""
상담사가 답변하는 방식으로 하지 말고, 고객의 음성을 변환하는 것에 초점을 맞춰서
다음 문장을 고객의 말투를 유지하면서, 욕설 및 무례한 표현을 제거하고 공손한 표현으로 바꿔줘:
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": ""}, 
            {"role": "user", "content": prompt + text}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def synthesize_audio(text: str) -> bytes:
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code="ko-KR", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    return response.audio_content



st.markdown("""
<style>
/* … 이미 있는 CSS … */

/* --- Title gradient only --------------------------------- */
h1.gradient {
    font-size: 2.5rem;
    font-weight: 600;
    text-align: center;
    background: linear-gradient(90deg,#007AFF 0%,#34C759 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='gradient'>🎧 soft 혐오표현 분류기</h1>", unsafe_allow_html=True)

with st.expander("ℹ️ 서비스 개요", expanded=True):
    st.markdown("""
    이 애플리케이션은 업로드된 음성 파일 또는 마이크 녹음에서 **욕설 및 혐오 표현**을 탐지하고,
    필요 시 **순화된 문장으로 TTS 합성**까지 진행합니다.
    """)

EXTENDED_FILE = "data/extended_bad_words.txt"

# ─────────────────────────────────────────────
# 1. 음성 입력 구역
with st.container():
    st.markdown(
        "<div style='background-color:#fff; padding:18px 18px 8px 18px; border-radius:10px; margin-bottom:12px; border:1px solid #eee;'>"
        "<b>1️⃣ 음성 입력</b><br>"
        "<span style='color:#666;'>마이크로 직접 녹음하거나, 음성 파일을 업로드하세요.</span>"
        "</div>", unsafe_allow_html=True
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        audio = mic_recorder(
            start_prompt="🎤 녹음 시작", 
            stop_prompt="⏹️ 녹음 종료", 
            key="recorder"
        )
    with col2:
        uploaded_file = st.file_uploader("또는 음성 파일 업로드 (webm/ogg/wav)", type=["webm", "ogg", "wav"])

    audio_bytes = None
    if audio:
        st.audio(audio['bytes'], format='audio/webm')
        audio_bytes = audio['bytes']
    elif uploaded_file:
        st.audio(uploaded_file, format='audio/wav')
        audio_bytes = uploaded_file.read()

# ─────────────────────────────────────────────
# 2. 음성 → 텍스트 변환 & 탐지
if audio_bytes:
    with st.container():
        st.markdown(
            "<div style='background-color:#fff; padding:18px 18px 8px 18px; border-radius:10px; margin-bottom:12px; border:1px solid #eee;'>"
            "<b>2️⃣ 음성 → 텍스트 변환</b><br>"
            "<span style='color:#666;'>Google Speech-to-Text로 음성을 텍스트로 변환합니다.</span>"
            "</div>", unsafe_allow_html=True
        )

        with st.spinner("음성 처리 준비 중..."):
            speech_client = load_speech_client()
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
                sample_rate_hertz=48000,
                language_code="ko-KR",
                enable_automatic_punctuation=True
            )
            try:
                response = speech_client.recognize(
                    config=config,
                    audio=speech.RecognitionAudio(content=audio_bytes)
                )
            except Exception as e:
                st.error(f"음성 인식 중 오류: {e}")
                st.stop()
            transcript = " ".join([res.alternatives[0].transcript for res in response.results]).strip()
            st.code(transcript, language="text")

    # ─────────────────────────────────────────────
    # 3. 욕설/혐오 탐지 카드
    with st.container():
        st.markdown(
            "<div style='background-color:#fff; padding:18px 18px 8px 18px; border-radius:10px; margin-bottom:12px; border:1px solid #eee;'>"
            "<b>3️⃣ 욕설/혐오 탐지</b><br>"
            "<span style='color:#666;'>욕설 또는 혐오 표현이 포함되어 있는지 확인합니다.</span>"
            "</div>", unsafe_allow_html=True
        )

        swears = contains_bad_word_loose(transcript, load_bad_words(EXTENDED_FILE))
        cleaned = transcript
        is_hate = False
        if swears:
            st.warning("⚠️ **욕설 발견됨**")
            cleaned = sanitize_text(transcript)
        else:
            st.info("✅ 욕설 없음. 혐오 표현 분석으로 넘어갑니다.")
            hate_pipe = load_hate_pipeline("model")
            label = hate_pipe(transcript)[0]
            is_hate = label['label'] == 'LABEL_1'
            if is_hate:
                st.warning("⚠️ **혐오 표현 감지됨**")
                cleaned = sanitize_text(transcript)
            else:
                st.success("✅ 혐오 표현도 발견되지 않음. 원문 그대로 사용됩니다.")

    # ─────────────────────────────────────────────
    # 4. 최종 출력 및 TTS 카드
    with st.container():
        st.markdown(
            "<div style='background-color:#fff; padding:18px 18px 8px 18px; border-radius:10px; margin-bottom:20px; border:1px solid #eee;'>"
            "<b>4️⃣ 최종 문장 및 음성 합성</b><br>"
            "<span style='color:#666;'>최종 문장을 확인하고, 음성으로 들어보세요.</span>"
            "</div>",
            unsafe_allow_html=True
        )

        # ✅ 최종 문장 강조 및 여백 추가
        st.markdown(
            f"<div style='font-size:1.3rem; font-weight:600; color:#222; margin:20px 0 30px 0;'>"
            f"➡️ {cleaned}</div>",
            unsafe_allow_html=True
        )

        audio_bytes_out = synthesize_audio(cleaned)

        # ✅ 오디오와 다운로드 버튼은 여백을 두고 아래에 배치
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
            st.audio(audio_bytes_out, format='audio/mp3')
        with col2:
            st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
            st.download_button(
                label="⬇️ MP3 다운로드",
                data=audio_bytes_out,
                file_name="output.mp3",
                mime="audio/mp3"
            )
