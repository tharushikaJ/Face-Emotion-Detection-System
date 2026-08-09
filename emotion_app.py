import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os
import h5py
import json

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Face Emotion Detector",
    page_icon="😊",
    layout="centered"
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e0e0e0;
    }
    .main-header {
        text-align: center;
        padding: 2rem 1rem 1rem;
    }
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffffff;
    }
    .main-header p {
        color: #a0aec0;
        font-size: 1rem;
    }
    .card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(10px);
    }
    .emotion-result {
        text-align: center;
        padding: 2rem;
        border-radius: 16px;
        margin: 1rem 0;
        font-size: 1.1rem;
    }
    .confidence-bar-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        margin-bottom: 2px;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem;
        font-size: 1rem;
        font-weight: 700;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Emotion Config
# ============================================================
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']

EMOTION_CONFIG = {
    'Angry':     {'emoji': '😠', 'color': '#fc8181', 'bg': 'rgba(252,129,129,0.15)', 'border': '#fc8181'},
    'Disgust':   {'emoji': '🤢', 'color': '#68d391', 'bg': 'rgba(104,211,145,0.15)', 'border': '#68d391'},
    'Fear':      {'emoji': '😨', 'color': '#b794f4', 'bg': 'rgba(183,148,244,0.15)', 'border': '#b794f4'},
    'Happy':     {'emoji': '😊', 'color': '#fbd38d', 'bg': 'rgba(251,211,141,0.15)', 'border': '#fbd38d'},
    'Neutral':   {'emoji': '😐', 'color': '#90cdf4', 'bg': 'rgba(144,205,244,0.15)', 'border': '#90cdf4'},
    'Sad':       {'emoji': '😢', 'color': '#63b3ed', 'bg': 'rgba(99,179,237,0.15)',  'border': '#63b3ed'},
    'Surprised': {'emoji': '😲', 'color': '#f6e05e', 'bg': 'rgba(246,224,94,0.15)',  'border': '#f6e05e'},
}

TIPS = {
    'Angry':     "Take a deep breath. Step away from the situation and give yourself time to calm down.",
    'Disgust':   "It's okay to feel this way. Try to shift focus to something pleasant.",
    'Fear':      "You are safe. Try grounding techniques — focus on 5 things you can see around you.",
    'Happy':     "Great to see you're happy! Keep spreading that positivity 🌟",
    'Neutral':   "You look calm and composed. A great state for focused work!",
    'Sad':       "It's okay to feel sad. Talk to someone you trust — you're not alone 💙",
    'Surprised': "Something caught you off guard! Take a moment to process it.",
}

# ============================================================
# Load Model
# ============================================================
@st.cache_resource
def load_model():
    model_path = 'emotion_model.h5'
    if not os.path.exists(model_path):
        return None

    # Patch: remove unsupported 'quantization_config' key
    with h5py.File(model_path, 'r+') as f:
        model_config = json.loads(f.attrs['model_config'])

        def remove_key(obj, key):
            if isinstance(obj, dict):
                obj.pop(key, None)
                for v in obj.values():
                    remove_key(v, key)
            elif isinstance(obj, list):
                for item in obj:
                    remove_key(item, key)

        remove_key(model_config, 'quantization_config')
        f.attrs['model_config'] = json.dumps(model_config)

    return tf.keras.models.load_model(model_path, compile=False)


# ============================================================
# Face Detection + Prediction
# ============================================================
def detect_and_predict(image, model):
    img_array = np.array(image)

    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_array

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    results = []
    img_display = img_bgr.copy()

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (48, 48))
        face_normalized = face_resized / 255.0
        face_input = face_normalized.reshape(1, 48, 48, 1)

        predictions = model.predict(face_input, verbose=0)[0]
        emotion_idx = np.argmax(predictions)
        emotion = EMOTIONS[emotion_idx]
        confidence = predictions[emotion_idx] * 100

        color_bgr = (100, 200, 100)
        cv2.rectangle(img_display, (x, y), (x+w, y+h), color_bgr, 2)
        label = f"{EMOTION_CONFIG[emotion]['emoji']} {emotion} {confidence:.0f}%"
        cv2.putText(img_display, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)

        results.append({
            'emotion': emotion,
            'confidence': confidence,
            'all_predictions': predictions * 100
        })

    img_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
    return img_rgb, results, len(faces)


# ============================================================
# UI
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>😊 Face Emotion Detector</h1>
    <p>Upload a face photo and let AI detect the emotion using Deep Learning (CNN)</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

model = load_model()

if model is None:
    st.markdown("""
    <div style="background:rgba(252,129,129,0.15); border:1px solid #fc8181;
                border-radius:12px; padding:1.2rem; margin-bottom:1rem;">
        <h4 style="color:#fc8181;">⚠️ Model Not Found</h4>
        <p>Place <strong>emotion_model.h5</strong> in the same folder as app.py, then restart the app.</p>
        <p style="font-size:0.85rem; color:#a0aec0;">Train the model using the provided <strong>train_model.ipynb</strong> in Google Colab.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

st.success("✅ Model loaded successfully!")

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 📷 Upload a Face Image")
st.markdown("Supported formats: JPG, JPEG, PNG")
uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Image**")
        st.image(image, use_column_width=True)

    with st.spinner("🔍 Detecting faces and analyzing emotions..."):
        result_img, results, face_count = detect_and_predict(image, model)

    with col2:
        st.markdown("**Detection Result**")
        st.image(result_img, use_column_width=True)

    st.markdown("---")

    if face_count == 0:
        st.warning("😕 No face detected. Please upload a clear front-facing photo.")

    else:
        st.markdown(f"### 🎯 Detected {face_count} face(s)")

        for i, result in enumerate(results):
            emotion = result['emotion']
            confidence = result['confidence']
            cfg = EMOTION_CONFIG[emotion]

            st.markdown(f"""
            <div class="emotion-result" style="background:{cfg['bg']}; border: 2px solid {cfg['border']};">
                <div style="font-size:3.5rem;">{cfg['emoji']}</div>
                <div style="font-size:1.8rem; font-weight:800; color:{cfg['color']}; margin: 0.5rem 0;">
                    {emotion}
                </div>
                <div style="font-size:1rem; color:#e2e8f0;">
                    Confidence: <strong>{confidence:.1f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.info(f"💡 **Tip:** {TIPS[emotion]}")

            with st.expander(f"📊 Face {i+1} — Full Emotion Breakdown"):
                all_preds = result['all_predictions']
                sorted_idx = np.argsort(all_preds)[::-1]

                for idx in sorted_idx:
                    emo = EMOTIONS[idx]
                    prob = all_preds[idx]
                    ecfg = EMOTION_CONFIG[emo]
                    st.markdown(f"""
                    <div style="margin-bottom:8px;">
                        <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:3px;">
                            <span>{ecfg['emoji']} {emo}</span>
                            <span style="color:{ecfg['color']}">{prob:.1f}%</span>
                        </div>
                        <div style="background:rgba(255,255,255,0.1); border-radius:6px; height:8px;">
                            <div style="background:{ecfg['color']}; width:{prob}%; height:8px; border-radius:6px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="background:rgba(255,200,0,0.08); border:1px solid rgba(255,200,0,0.3);
                    border-radius:10px; padding:0.9rem; font-size:0.82rem; color:#faf089;">
            ⚠️ <strong>Note:</strong> This is an AI prediction for educational purposes.
            Emotion detection accuracy depends on image quality, lighting, and face angle.
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#718096; font-size:0.8rem; padding-bottom:1rem;">
    Powered by CNN (Convolutional Neural Network) · FER-2013 Dataset · OpenCV Face Detection<br>
    Emotions: Angry · Disgust · Fear · Happy · Neutral · Sad · Surprised
</div>
""", unsafe_allow_html=True)