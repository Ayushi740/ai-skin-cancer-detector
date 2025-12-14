import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

from src.gradcam import generate_gradcam
from src.report import generate_pdf_report

from disease_info import disease_details

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "models/skin_cancer_model.h5"
EXPLAIN_DIR = "explainability"
IMAGE_SIZE = (224, 224)

os.makedirs(EXPLAIN_DIR, exist_ok=True)

CLASS_NAMES = [
    "Actinic Keratosis",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Nevus",
    "Vascular Lesion"
]

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="AI Skin Cancer Detector",
    page_icon="🧬",
    layout="wide"
)

# ==============================
# CUSTOM MEDICAL UI CSS
# ==============================
st.markdown("""
<style>
body { background-color: #f6fbff; }

.main-card {
    background: white;
    padding: 30px;
    border-radius: 18px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.06);
    margin-bottom: 25px;
}

.title {
    font-size: 44px;
    font-weight: 800;
    color: #0b3c5d;
    text-align: center;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #555;
    margin-bottom: 25px;
}

.section-title {
    font-size: 26px;
    font-weight: 700;
    color: #0b3c5d;
    margin-bottom: 15px;
}

.info-card {
    background-color: #f9fcff;
    padding: 18px;
    border-radius: 14px;
    border-left: 6px solid #2196f3;
    margin-bottom: 12px;
}

.prediction-box {
    background: linear-gradient(135deg, #e3f2fd, #bbdefb);
    padding: 22px;
    border-radius: 16px;
    font-size: 22px;
    font-weight: bold;
    color: #0d47a1;
    text-align: center;
}

.confidence-box {
    background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
    padding: 22px;
    border-radius: 16px;
    font-size: 22px;
    font-weight: bold;
    color: #1b5e20;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# HELPER FUNCTIONS
# ==============================
def preprocess_for_model(image):
    img = image.resize(IMAGE_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict(image):
    arr = preprocess_for_model(image)
    preds = model.predict(arr, verbose=0)[0]
    idx = np.argmax(preds)
    return CLASS_NAMES[idx], float(preds[idx])

# ==============================
# SIDEBAR
# ==============================
st.sidebar.markdown("## 🧭 Navigation")
page = st.sidebar.radio(
    "Select Section",
    ["🏠 Home", "🔍 AI Prediction", "🧪 Explainability (Grad-CAM)", "📄 Download Report", "ℹ About"]
)

# ==============================
# HOME PAGE
# ==============================
if page == "🏠 Home":
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)

    st.markdown("<h1 class='title'>🧬 AI Skin Cancer Detector</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle'>Deep learning powered skin lesion analysis with explainable AI</p>",
        unsafe_allow_html=True
    )

    

    # ===== SYSTEM CAPABILITIES (ATTRACTIVE) =====
    st.markdown("<div class='section-title'>🚀 System Capabilities</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        🧬 <b>Multi-Class Skin Lesion Detection</b><br>
        Accurately identifies <b>7 different types of skin lesions</b> using a deep learning model.
    </div>

    <div class="info-card">
        📊 <b>Prediction Confidence Score</b><br>
        Displays probability score to assess reliability of the AI prediction.
    </div>

    <div class="info-card">
        🧠 <b>Explainable AI using Grad-CAM</b><br>
        Highlights important regions of the image that influenced the model decision.
    </div>

    <div class="info-card">
        📄 <b>Automated Medical PDF Report</b><br>
        Generates a structured, professional medical report instantly.
    </div>

    <div class="info-card">
        🏥 <b>Clinical-Grade User Interface</b><br>
        Clean, hospital-style dashboard suitable for doctors and students.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# AI PREDICTION PAGE
# ==============================
if page == "🔍 AI Prediction":
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🔍 Upload Skin Lesion Image</div>", unsafe_allow_html=True)

    file = st.file_uploader(
        "Supported formats: JPG, JPEG, PNG",
        type=["jpg", "jpeg", "png"]
    )

    if file:
        image = Image.open(file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🧠 Run AI Prediction"):
            with st.spinner("Analyzing image..."):
                label, conf = predict(image)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"<div class='prediction-box'>🩺 Prediction<br>{label}</div>",
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    f"<div class='confidence-box'>📊 Confidence<br>{conf:.2f}</div>",
                    unsafe_allow_html=True
                )

            info = disease_details[label]

            st.markdown("<div class='section-title'>📌 Condition Information</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='info-card'><b>Severity:</b> {info['severity']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='info-card'><b>Description:</b> {info['description']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='info-card'><b>Medical Advice:</b> {info['advice']}</div>", unsafe_allow_html=True)

            st.session_state["last_image"] = image
            st.session_state["last_label"] = label
            st.session_state["last_conf"] = conf

            st.warning("⚠ Educational use only. Not a medical diagnosis.")

    st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# GRAD-CAM PAGE
# ==============================
if page == "🧪 Explainability (Grad-CAM)":
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🧪 Model Explainability</div>", unsafe_allow_html=True)

    if "last_image" not in st.session_state:
        st.info("Please perform a prediction first.")
    else:
        img_array = preprocess_for_model(st.session_state["last_image"])
        heatmap = generate_gradcam(model, img_array)

        st.image(
            heatmap,
            caption=f"Grad-CAM Heatmap — {st.session_state['last_label']}",
            use_container_width=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# PDF REPORT PAGE
# ==============================
if page == "📄 Download Report":
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📄 Medical Report</div>", unsafe_allow_html=True)

    if "last_image" not in st.session_state:
        st.info("Run a prediction first to generate the report.")
    else:
        if st.button("📑 Generate PDF Report"):
            with st.spinner("Generating report..."):
                pdf_path = generate_pdf_report(
                    st.session_state["last_image"],
                    st.session_state["last_label"],
                    st.session_state["last_conf"]
                )

            with open(pdf_path, "rb") as f:
                st.download_button(
                    "📥 Download AI Medical Report",
                    f,
                    file_name="AI_Skin_Cancer_Report.pdf"
                )

    st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# ABOUT PAGE
# ==============================
if page == "ℹ About":
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>ℹ About This Project</div>", unsafe_allow_html=True)

    st.write("""
    **AI Skin Cancer Detector** is a deep learning-based diagnostic support system.

    **Technologies Used**
    - TensorFlow / Keras (MobileNetV2)
    - Streamlit
    - Grad-CAM Explainable AI
    - Python, NumPy, PIL
    """)

    st.markdown("</div>", unsafe_allow_html=True)
