import streamlit as st
import numpy as np
import pickle
import librosa
import librosa.display
import soundfile as sf
from io import BytesIO
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
st.set_page_config(page_title="🎤 Emotion Detection from Voice", layout="centered")

# ---------- HEADER ----------
st.markdown(
    """
    <style>
    .logo-container {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    .logo-container img {
        height: 60px;
        margin-right: 15px;
    }
    </style>
    <div class="logo-container">
        <img src="https://cdn-icons-png.flaticon.com/512/3208/3208707.png" />
        <h2>🎤 Emotion Detection from Voice</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    with open(r"D:\ML PROJECTS\Emotion Detection from Voice\emotion_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(r"D:\ML PROJECTS\Emotion Detection from Voice\label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, encoder

model, encoder = load_model()

# ---------- FEATURE EXTRACTION ----------
def extract_mfcc_features(audio_stream):
    try:
        y, sr = sf.read(audio_stream)
        if len(y.shape) > 1:
            y = y.mean(axis=1)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return np.mean(mfcc.T, axis=0), y, sr
    except Exception as e:
        st.error(f"❌ Error extracting features: {e}")
        return None, None, None

# ---------- FILE UPLOAD ----------
st.markdown("### 🎙️ Upload a WAV audio file:")
uploaded_file = st.file_uploader("Upload .wav file here", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    audio_bytes = uploaded_file.read()
    audio_buffer = BytesIO(audio_bytes)

    st.markdown("🔍 Extracting features...")
    features, y, sr = extract_mfcc_features(audio_buffer)

    if features is not None and features.shape[0] == 40:
        features = features.reshape(1, -1)

        # ---------- EMOTION PREDICTION ----------
        prediction = model.predict(features)
        emotion = encoder.inverse_transform(prediction)[0]
        st.success(f"✅ Predicted Emotion: **{emotion}**")

        # ---------- PROBABILITIES ----------
        st.markdown("### 📊 Prediction Probabilities:")
        probs = model.predict_proba(features)[0]
        emotions = encoder.inverse_transform(np.arange(len(probs)))

        fig1, ax1 = plt.subplots()
        bars = ax1.bar(emotions, probs, color='skyblue')
        ax1.set_ylim([0, 1])
        ax1.set_ylabel("Probability")
        ax1.set_title("Emotion Prediction Probabilities")
        ax1.bar_label(bars, fmt='%.2f', padding=3)
        st.pyplot(fig1)

        # ---------- WAVEFORM ----------
        st.markdown("### 🔉 Waveform")
        fig2, ax2 = plt.subplots()
        librosa.display.waveshow(y, sr=sr, ax=ax2)
        ax2.set_title("Waveform")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude")
        st.pyplot(fig2)

        # ---------- SPECTROGRAM ----------
        st.markdown("### 🌈 Mel Spectrogram")
        fig3, ax3 = plt.subplots()
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_DB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel', cmap='magma', ax=ax3)
        ax3.set_title("Mel Spectrogram")
        fig3.colorbar(img, ax=ax3, format='%+2.0f dB')
        st.pyplot(fig3)

    else:
        st.warning("⚠️ Feature extraction failed or unsupported format.")

