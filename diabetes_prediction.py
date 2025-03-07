import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("diabetes_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the StandardScaler (if used in training)
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

st.title("Diabetes Prediction App")
st.write("Enter the patient details to predict diabetes.")

# Collect user inputs
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=80)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=30)
bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=0, max_value=120, value=30)

# Predict button
if st.button("Predict Diabetes"):
    # Create feature array
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    
    # Scale the features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)
    
    # Display result
    result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    st.subheader(f"Prediction: {result}")

    # Show probability score
    probability = model.predict_proba(features_scaled)[0][1] * 100
    st.write(f"Probability of being diabetic: {probability:.2f}%")
