import os
import tempfile
import streamlit as st
import openai
from transformers import pipeline
from google.cloud import texttospeech, speech
import librosa
from dotenv import load_dotenv
import soundfile as sf

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "textmining-461305-f0cddebc86fe.json"

# 1. STT 클라이언트 로드
@st.cache_resource
def load_speech_client():
    return speech.SpeechClient()

# 2. 룰베이스 욕설 사전 로드
@st.cache_resource
def load_bad_words(filepath: str):
    words = []
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            words += [w.strip().lower() for w in f if w.strip()]
    return set(words)

# 3. 욕설 탐지 함수
def contains_bad_word_loose(text: str, bad_words: set) -> set:
    lowered = text.lower()
    return {bw for bw in bad_words if bw and bw in lowered}

# 4. 혐오 탐지 파이프라인 로드
@st.cache_resource
def load_hate_pipeline(model_path: str):
    return pipeline("text-classification", model=model_path, tokenizer=model_path)

# 5. 순화어 변환 (GPT API 호출)
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

# 6. TTS (Google Cloud)
def synthesize_audio(text: str) -> bytes:
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="ko-KR",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    return response.audio_content

# Streamlit UI
st.title("음성 프로세싱 스트림릿 프로토타입")

# 설정
EXTENDED_FILE = "extended_bad_words.txt"

# 업로드된 오디오 파일 처리
audio_file = st.file_uploader("오디오 파일 업로드 (.wav, .mp3)", type=["wav", "mp3"])
if audio_file:
    # STT 준비
    speech_client = load_speech_client()
    bad_words = load_bad_words(EXTENDED_FILE)
    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(audio_file.name)[1], delete=False) as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    # 1) STT: librosa + soundfile 로 리샘플링
    st.info("음성 인식 중 (Google Cloud Speech-to-Text)...")
    # librosa로 로드, 16kHz mono로 리샘플링
    y, sr = librosa.load(tmp_path, sr=16000, mono=True)
    converted = tmp_path + ".wav"
    # PCM 16bit WAV로 저장
    sf.write(converted, y, 16000, format='WAV', subtype='PCM_16')

    with open(converted, "rb") as f:
        audio_content = f.read()
    audio = speech.RecognitionAudio(content=audio_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="ko-KR",
        enable_automatic_punctuation=True
    )
    response = speech_client.recognize(config=config, audio=audio)
    transcript = " ".join([res.alternatives[0].transcript for res in response.results]).strip()
    st.success(f"인식 결과: {transcript}")

    # 2) 욕설 탐지
    swears = contains_bad_word_loose(transcript, bad_words)
    has_swear = bool(swears)
    st.write(f"욕설 탐지: {has_swear} {list(swears)}")

    # 3) 흐름 분기
    if has_swear:
        st.info("순화어 변환 중 (욕설)...")
        cleaned = sanitize_text(transcript)
        st.write(f"순화된 문장: {cleaned}")
    else:
        hate_pipe = load_hate_pipeline("model")
        label = hate_pipe(transcript)[0]
        is_hate = label['label'] == 'LABEL_1'
        st.write(f"혐오 표현: {is_hate} (스코어: {label['score']:.2f})")
        if is_hate:
            st.info("순화어 변환 중 (혐오)...")
            cleaned = sanitize_text(transcript)
            st.write(f"순화된 문장: {cleaned}")
        else:
            cleaned = transcript
            st.write("원본 문장 사용")

    # 4) TTS 합성
    st.info("음성 합성 중...")
    audio_bytes = synthesize_audio(cleaned)
    st.audio(audio_bytes, format='audio/mp3')
    st.success("완료")

    # 5) 다운로드 버튼
    st.download_button(
        label="MP3 다운로드",
        data=audio_bytes,
        file_name="output.mp3",
        mime="audio/mp3"
    )