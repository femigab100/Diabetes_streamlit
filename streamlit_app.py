# streamlit_app.py

import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open('diabetes_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Streamlit UI
st.set_page_config(page_title="Diabetes Predictor", layout="centered")
st.title("🩺 Diabetes Prediction App")
st.write("Enter patient health metrics to predict diabetic status.")

# Input fields
pregnancies = st.slider("Pregnancies", 0, 20, 1)
glucose = st.slider("Glucose Level", 50, 200, 100)
blood_pressure = st.slider("Blood Pressure", 40, 120, 70)
skin_thickness = st.slider("Skin Thickness", 10, 99, 20)
insulin = st.slider("Insulin Level", 15, 276, 80)
bmi = st.slider("BMI", 15.0, 50.0, 25.0)
diabetes_pedigree = st.slider("Diabetes Pedigree Function", 0.1, 2.5, 0.5)
age = st.slider("Age", 21, 81, 35)

# Predict button
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, diabetes_pedigree, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Patient is likely diabetic. Risk score: {probability:.2f}")
    else:
        st.success(f"✅ Patient is likely non-diabetic. Risk score: {probability:.2f}")