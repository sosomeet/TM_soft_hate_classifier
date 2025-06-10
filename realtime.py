import os
import streamlit as st
import openai
from transformers import pipeline
from google.cloud import texttospeech, speech
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder

st.set_page_config(page_title="✨ 음성 클렌징 & 합성 데모", layout="centered")

# ──────────────────────────────────────────────────────
# API 인증 및 클라이언트 설정
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "textmining-461305-f0cddebc86fe.json"

# ──────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────
# 핵심 처리 함수
def contains_bad_word_loose(text: str, bad_words: set) -> set:
    return {bw for bw in bad_words if bw and bw in text.lower()}

def sanitize_text(text: str) -> str:
    prompt = f"""
다음 문장을 고객의 말투를 유지하면서, 욕설 및 무례한 표현을 제거하고 공손한 표현으로 바꿔줘:
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": """
    """}, 
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

# ──────────────────────────────────────────────────────
# Streamlit UI
st.title("🎧 음성 정제 & TTS 변환기")

with st.expander("ℹ️ 서비스 개요", expanded=True):
    st.markdown("""
    이 애플리케이션은 업로드된 음성 파일 또는 마이크 녹음에서 **욕설 및 혐오 표현**을 탐지하고,
    필요 시 **순화된 문장으로 TTS 합성**까지 진행합니다.
    """)

EXTENDED_FILE = "extended_bad_words.txt"

# 1. 마이크 녹음 버튼 추가
audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="⏹️ 녹음 종료", key="recorder")

if audio:
    # 브라우저 녹음은 보통 webm/opus 포맷이므로, audio/wav 대신 audio/webm 또는 audio/ogg로 재생
    st.audio(audio['bytes'], format='audio/webm')  # 또는 format='audio/ogg' (브라우저 녹음 포맷에 따라 다름)

    with st.spinner("음성 처리 준비 중..."):
        # 파일 저장은 안 해도 되지만, 필요하다면 임시 저장
        # (여기서는 변환 없이 바로 STT에 전송)

        # Google STT 클라이언트 생성
        speech_client = load_speech_client()

        # WebM_OPUS 또는 OGG_OPUS로 설정 (브라우저 녹음은 보통 48000Hz)
        # 실제 녹음 포맷이 webm/opus인지 확인 필요
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,  # 또는 OGG_OPUS
            sample_rate_hertz=48000,  # 브라우저 기본 샘플링
            language_code="ko-KR",
            enable_automatic_punctuation=True
        )

        # 오디오 데이터 전송
        audio_content = audio['bytes']
        try:
            response = speech_client.recognize(
                config=config,
                audio=speech.RecognitionAudio(content=audio_content)
            )
        except Exception as e:
            st.error(f"음성 인식 중 오류: {e}")
            st.stop()

        # 결과 출력
        transcript = " ".join([res.alternatives[0].transcript for res in response.results]).strip()
        st.code(transcript, language="text")

    # ────────────────────────────────
    # 욕설/혐오 탐지 및 순화, TTS (기존 코드와 동일)
    swears = contains_bad_word_loose(transcript, load_bad_words(EXTENDED_FILE))
    if swears:
        st.warning(f"⚠️ 욕설 발견됨: {', '.join(swears)}")
        cleaned = sanitize_text(transcript)
    else:
        st.info("✅ 욕설 없음. 혐오 표현 분석으로 넘어갑니다.")
        hate_pipe = load_hate_pipeline("model")
        label = hate_pipe(transcript)[0]
        is_hate = label['label'] == 'LABEL_1'
        if is_hate:
            st.warning(f"⚠️ 혐오 표현 감지됨 (신뢰도: {label['score']:.2f})")
            cleaned = sanitize_text(transcript)
        else:
            st.success("✅ 혐오 표현도 발견되지 않음. 원문 그대로 사용됩니다.")
            cleaned = transcript

    st.subheader("📝 최종 출력 문장")
    st.markdown(f"**➡️ {cleaned}**")

    st.subheader("🔈 음성 합성 결과")
    audio_bytes = synthesize_audio(cleaned)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.audio(audio_bytes, format='audio/mp3')
    with col2:
        st.download_button(
            label="⬇️ MP3 다운로드",
            data=audio_bytes,
            file_name="output.mp3",
            mime="audio/mp3"
        )
