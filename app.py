import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from PIL import Image

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Healthcare AI Intelligence Suite",
    page_icon="🏥",
    layout="wide"
)

# =====================================================
# PREMIUM CSS (UPGRADED)
# =====================================================
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Glass Card */
.card {
    padding: 25px;
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(15px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.37);
    margin-bottom: 25px;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    font-weight: bold;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #0072ff, #00c6ff);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #141e30, #243b55);
    color: white;
}

/* Metrics */
[data-testid="stMetricValue"] {
    font-size: 28px;
    font-weight: bold;
}

/* Headers */
h1, h2, h3 {
    color: #00e5ff;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODELS
# =====================================================
@st.cache_resource
def load_models():
    disease_model = joblib.load("models/disease_model.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    heart_model = joblib.load("models/risk_model.pkl")
    xray_model = tf.keras.models.load_model("models/xray_model.h5")
    return disease_model, label_encoder, heart_model, xray_model

disease_model, label_encoder, heart_model, xray_model = load_models()

# =====================================================
# PREMIUM HEADER
# =====================================================
st.markdown("""
<div style='text-align:center; padding: 20px;'>
    <h1>🏥 AI Healthcare Intelligence Suite</h1>
    <p style='font-size:18px; color:lightgray;'>
        Advanced Multi-Model Diagnostic System powered by Machine Learning & Deep Learning
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =====================================================
# SIDEBAR
# =====================================================
menu = st.sidebar.selectbox(
    "🔍 Choose Diagnostic Module",
    ["🩺 Symptom Disease Prediction",
     "❤️ Heart Disease Risk",
     "🫁 Chest X-Ray Analysis"]
)

# =====================================================
# 🩺 SYMPTOM DISEASE PREDICTION
# =====================================================
if menu == "🩺 Symptom Disease Prediction":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("🩺 Symptom Based Disease Diagnosis")

    data = pd.read_csv("dataset/symptoms/dataset.csv")
    symptom_columns = data.columns[:-1]

    input_data = []
    col1, col2 = st.columns(2)

    for i, symptom in enumerate(symptom_columns):
        if i % 2 == 0:
            val = col1.selectbox(symptom, [0, 1], key=symptom)
        else:
            val = col2.selectbox(symptom, [0, 1], key=symptom)
        input_data.append(val)

    if st.button("Predict Disease"):

        input_df = pd.DataFrame([input_data], columns=symptom_columns)
        prediction = disease_model.predict(input_df)
        probabilities = disease_model.predict_proba(input_df)

        disease_name = label_encoder.inverse_transform(prediction)[0]
        confidence = np.max(probabilities) * 100

        st.markdown(
            f"<div style='padding:15px; border-radius:10px; background:rgba(0,255,255,0.15);'>"
            f"🩺 <b>Predicted Disease: {disease_name}</b>"
            "</div>", unsafe_allow_html=True
        )

        st.metric("Confidence Level", f"{confidence:.2f}%")
        st.progress(int(confidence))

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# ❤️ HEART DISEASE RISK
# =====================================================
elif menu == "❤️ Heart Disease Risk":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("❤️ Heart Disease Risk Prediction")

    col1, col2 = st.columns(2)

    age = col1.slider("Age", 20, 100, 40)
    sex = col2.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    cp = col1.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
    trestbps = col2.slider("Resting Blood Pressure", 80, 200, 120)
    chol = col1.slider("Cholesterol", 100, 400, 200)
    fbs = col2.selectbox("Fasting Blood Sugar > 120", [0,1])
    restecg = col1.selectbox("Resting ECG (0-2)", [0,1,2])
    thalach = col2.slider("Max Heart Rate", 60, 220, 150)
    exang = col1.selectbox("Exercise Induced Angina", [0,1])
    oldpeak = col2.slider("ST Depression", 0.0, 6.0, 1.0)
    slope = col1.selectbox("Slope (0-2)", [0,1,2])
    ca = col2.selectbox("Major Vessels (0-3)", [0,1,2,3])
    thal = col1.selectbox("Thalassemia (0-3)", [0,1,2,3])

    if st.button("Analyze Heart Risk"):

        input_df = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs,
                                  restecg, thalach, exang, oldpeak,
                                  slope, ca, thal]],
                                columns=['age','sex','cp','trestbps','chol','fbs',
                                         'restecg','thalach','exang','oldpeak',
                                         'slope','ca','thal'])

        prediction = heart_model.predict(input_df)
        probabilities = heart_model.predict_proba(input_df)
        confidence = np.max(probabilities) * 100

        if prediction[0] == 1:
            st.markdown(
                "<div style='padding:15px; border-radius:10px; background:rgba(255,0,0,0.25);'>"
                "⚠ <b>High Risk of Heart Disease Detected</b>"
                "</div>", unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div style='padding:15px; border-radius:10px; background:rgba(0,255,0,0.25);'>"
                "✅ <b>Low Risk of Heart Disease</b>"
                "</div>", unsafe_allow_html=True
            )

        st.metric("Model Confidence", f"{confidence:.2f}%")
        st.progress(int(confidence))

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# 🫁 X-RAY ANALYSIS
# =====================================================
elif menu == "🫁 Chest X-Ray Analysis":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("🫁 AI Powered Pneumonia Detection")

    uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file).resize((224, 224))
        st.image(image, caption="Uploaded X-Ray", use_column_width=True)

        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = xray_model.predict(img_array)
        confidence = float(prediction[0][0]) * 100

        if confidence > 50:
            st.markdown(
                "<div style='padding:15px; border-radius:10px; background:rgba(255,0,0,0.25);'>"
                "⚠ <b>Pneumonia Detected</b>"
                "</div>", unsafe_allow_html=True
            )
            st.metric("Confidence", f"{confidence:.2f}%")
            st.progress(int(confidence))
        else:
            st.markdown(
                "<div style='padding:15px; border-radius:10px; background:rgba(0,255,0,0.25);'>"
                "✅ <b>Normal Chest X-Ray</b>"
                "</div>", unsafe_allow_html=True
            )
            st.metric("Confidence", f"{100-confidence:.2f}%")
            st.progress(int(100-confidence))

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption("© 2026 AI Healthcare Intelligence Suite | Final Year Major Project | Built with Streamlit + ML + Deep Learning")
