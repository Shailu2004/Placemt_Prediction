import streamlit as st
import numpy as np
import pickle
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
import os

# ==========================================
# LOAD MODELS
# ==========================================

with open("placement_regression_model.pkl", "rb") as f:
    reg_model = pickle.load(f)

with open("placement_classification_model.pkl", "rb") as f:
    clf_model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# ==========================================
# GEMINI CONFIGURATION
# ==========================================

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(page_title="🎓 AI Placement Predictor", layout="wide")

st.title("🎓 AI-Powered Placement Prediction System")
st.markdown("Predict placement probability, category & get AI-driven career guidance.")

# ==========================================
# INPUT FORM
# ==========================================

with st.form("placement_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        tenth = st.number_input("10th Percentage", 40.0, 100.0, 85.0)
        twelfth = st.number_input("12th/Diploma Percentage", 40.0, 100.0, 80.0)
        cgpa = st.number_input("CGPA", 4.0, 10.0, 8.5)

    with col2:
        elq = st.number_input("AMCAT ELQ Score", 0, 100, 75)
        automata = st.number_input("AMCAT Automata Score", 0, 100, 70)
        leetcode = st.number_input("Leetcode Rating", 0, 3000, 1600)

    with col3:
        codeforces = st.number_input("Codeforces Rating", 0, 3500, 1700)
        passive = st.number_input("Passive Backlogs", 0, 10, 0)
        active = st.number_input("Active Backlogs", 0, 10, 0)

    submitted = st.form_submit_button("🔍 Predict Placement")

# ==========================================
# PREDICTION
# ==========================================

if submitted:

    input_df = pd.DataFrame([{
        "10th%": tenth,
        "12th/diploma%": twelfth,
        "CGPA": cgpa,
        "Amcat ELQ Score": elq,
        "Amcat Automata score": automata,
        "Leetcode rating": leetcode,
        "Codeforces rating": codeforces,
        "Passive backlog": passive,
        "Active backlog": active
    }])

    # SCALE INPUT (VERY IMPORTANT)
    scaled_input = scaler.transform(input_df)

    # Regression Prediction
    probability = reg_model.predict(scaled_input)[0]

    # Classification Prediction
    category_encoded = clf_model.predict(scaled_input)[0]
    category = le.inverse_transform([category_encoded])[0]

    st.success(f"🎯 Predicted Placement Probability: {probability:.2f}%")
    st.info(f"📊 Placement Category: {category}")

    # ==========================================
    # GEMINI PROMPT ENGINEERING
    # ==========================================

    prompt = f"""
You are an expert career placement advisor for engineering students.

Analyze the student's academic and coding profile below and provide structured guidance.

📌 Student Profile:
- 10th %: {tenth}
- 12th/Diploma %: {twelfth}
- CGPA: {cgpa}
- AMCAT ELQ: {elq}
- AMCAT Automata: {automata}
- Leetcode Rating: {leetcode}
- Codeforces Rating: {codeforces}
- Passive Backlogs: {passive}
- Active Backlogs: {active}

📊 Model Prediction:
- Placement Probability: {probability:.2f}%
- Placement Category: {category}

Respond STRICTLY in the following format:

🎓 Profile Evaluation:
(Short evaluation of strengths & weaknesses in bullet points)

🚀 Improvement Roadmap (Next 3 Months Plan):
(Actionable steps in bullet points)

💡 Skill Gap Analysis:
(Technical + Aptitude + Behavioral suggestions)

🔥 Final Verdict:
(Encouraging summary in 3-4 lines)

Keep tone professional, motivating, and precise.
Do not add any extra text outside this structure.
"""

    response = model.generate_content(prompt)

    st.markdown("### 🤖 AI Career Guidance")
    st.markdown(response.text)