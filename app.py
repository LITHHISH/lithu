import streamlit as st
import io
import torch
import torch.nn.functional as F
import librosa
import torchaudio
import numpy as np
import plotly.express as px
from tortoise.models.classifier import AudioMiniEncoderWithClassifierHead
from faster_whisper import WhisperModel

# ---------------------------
# Settings
# ---------------------------
MAX_DURATION = 5  # seconds to prevent memory crash
SAMPLING_RATE = 22000

# ---------------------------
# Load AI classifier (cached)
# ---------------------------
@st.cache_resource
def load_classifier():
    model = AudioMiniEncoderWithClassifierHead(
        2,
        spec_dim=1,
        embedding_dim=512,
        depth=5,
        downsample_factor=4,
        resnet_blocks=2,
        attn_blocks=4,
        num_attn_heads=4,
        base_channels=32,
        dropout=0,
        kernel_size=5,
        distribute_zero_label=False
    )
    state_dict = torch.load("classifier.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ---------------------------
# Load Whisper model (cached, tiny for speed)
# ---------------------------
@st.cache_resource
def load_whisper_model():
    return WhisperModel("tiny", device="cpu")

whisper_model = load_whisper_model()

# ---------------------------
# Detect language
# ---------------------------
LANG_MAP = {"en": "English", "ta": "Tamil", "hi": "Hindi", "te": "Telugu"}

def detect_language(file):
    segments, info = whisper_model.transcribe(file, beam_size=3)
    code = info.language
    return LANG_MAP.get(code, code)

# ---------------------------
# Load audio safely
# ---------------------------
def load_audio(file, sampling_rate=SAMPLING_RATE):
    if isinstance(file, io.BytesIO) or str(file).endswith(".mp3"):
        audio, sr = librosa.load(file, sr=sampling_rate)
        audio = torch.FloatTensor(audio)
    else:
        audio, sr = torchaudio.load(file)
        audio = audio[0]

    # Resample if needed
    if sr != sampling_rate:
        audio = torchaudio.functional.resample(audio, sr, sampling_rate)

    # Trim long audio
    max_samples = sampling_rate * MAX_DURATION
    if audio.shape[0] > max_samples:
        audio = audio[:max_samples]

    audio = audio.clamp(-1, 1)
    return audio.unsqueeze(0)

# ---------------------------
# Classify audio
# ---------------------------
def classify_audio_clip(model, clip):
    with torch.no_grad():
        clip = clip.unsqueeze(0)
        probs = F.softmax(model(clip), dim=-1)
    ai_prob = probs[0][0].item()
    human_prob = 1 - ai_prob
    conclusion = "Likely AI-generated voice" if ai_prob > 0.5 else "Likely human voice"
    return ai_prob, human_prob, conclusion

# ---------------------------
# Spam detection (based on transcription)
# ---------------------------
SPAM_KEYWORDS = ["win", "prize", "subscribe", "free", "offer", "click", "urgent", "money"]

def detect_spam(text):
    text_lower = text.lower()
    for keyword in SPAM_KEYWORDS:
        if keyword in text_lower:
            return "Spam"
    return "Not Spam"

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(layout="wide")
st.title("🎙️ AI Voice, Language & Spam Detection")

uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

if uploaded_file and st.button("Analyze Audio"):
    col1, col2, col3 = st.columns(3)

    # Column 1: Detection Result
    with col1:
        st.subheader("🔍 Detection Result")
        audio_clip = load_audio(uploaded_file)
        classifier = load_classifier()
        ai_prob, human_prob, conclusion = classify_audio_clip(classifier, audio_clip)
        language = detect_language(uploaded_file)

        # Transcribe for spam detection
        segments, info = whisper_model.transcribe(uploaded_file, beam_size=3)
        full_text = " ".join([segment.text for segment in segments])
        spam_result = detect_spam(full_text)

        st.metric(label="AI-Generated Probability", value=f"{ai_prob*100:.2f} %")
        st.metric(label="Human-Generated Probability", value=f"{human_prob*100:.2f} %")
        st.metric(label="Detected Language", value=language)
        st.metric(label="Spam Detection", value=spam_result)
        st.info(f"**Conclusion:** {conclusion}")

    # Column 2: Audio & Waveform
    with col2:
        st.subheader("🎧 Uploaded Audio")
        st.audio(uploaded_file)

        waveform = audio_clip.squeeze().numpy()
        fig = px.line(
            x=np.arange(len(waveform)),
            y=waveform,
            title="Waveform"
        )
        fig.update_layout(xaxis_title="Time", yaxis_title="Amplitude")
        st.plotly_chart(fig, use_container_width=True)

    # Column 3: Disclaimer
    with col3:
        st.subheader("⚠️ Disclaimer")
        st.warning(
            "This AI detection model is **not 100% accurate**.\n"
            "Spam detection is based on keywords and may not be perfect.\n"
            "Use results as a **strong signal**, not a final judgment.\n"
            "For best results, upload **short clips (~5 sec)**.\n"
            "Language detection may vary if audio is noisy or mixed."
        )
