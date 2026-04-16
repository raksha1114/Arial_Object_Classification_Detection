# ==========================================
# IMPORT LIBRARIES
# ==========================================

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from ultralytics import YOLO

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(page_title="Bird vs Drone Detection", layout="wide")

# ==========================================
# CSS FOR BIG HEADINGS
# ==========================================

st.markdown("""
<style>

/* Main Title */
.big-title {
    font-size: 55px !important;
    font-weight: 800 !important;
    text-align: center !important;
}

/* Section Headings */
.big-header {
    font-size: 30px !important;
    font-weight: 700 !important;
    margin-top: 20px !important;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# TITLE
# ==========================================

st.markdown('<h1 class="big-title">Aerial Object Classification & Detection</h1>', unsafe_allow_html=True)
st.write("Upload an image to classify and detect objects")
st.write("---")

# ==========================================
# LOAD MODELS
# ==========================================

clf_model = load_model("mobilenet_phase2.keras")
yolo_model = YOLO("best (1).pt")

# ==========================================
# FILE UPLOADER
# ==========================================

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# ==========================================
# MAIN LOGIC
# ==========================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    # 2 COLUMN LAYOUT
    col1, col2 = st.columns(2)

    # ======================================
    # LEFT SIDE → ONLY IMAGE
    # ======================================
    with col1:

        st.markdown('<h2 class="big-header">📷 Uploaded Image</h2>', unsafe_allow_html=True)
        st.image(image, width=450)

    # ======================================
    # RIGHT SIDE → CLASSIFICATION + DETECTION
    # ======================================
    with col2:

        # ================================
        # CLASSIFICATION
        # ================================
        st.markdown('<h2 class="big-header">🧠 Classification</h2>', unsafe_allow_html=True)

        # Preprocess
        img = image.resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = clf_model.predict(img)

        if prediction[0][0] > 0.5:
            label = " Drone"
            confidence = float(prediction[0][0])
        else:
            label = " Bird"
            confidence = float(1 - prediction[0][0])

        st.write(f"Prediction: **{label}**")

        # CONFIDENCE LINE (PROGRESS BAR)
        st.progress(confidence)

        st.write(f"Confidence: **{confidence:.2f}**")

        st.write("---")

        # ================================
        # OBJECT DETECTION
        # ================================
        st.markdown('<h2 class="big-header"> Object Detection</h2>', unsafe_allow_html=True)

        results = yolo_model(image, conf=0.15)
        result_img = results[0].plot()

        st.image(result_img, width=500)