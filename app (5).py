
import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import numpy as np

# Load the pre-trained scaler and model
scaler = joblib.load('scaler.pkl')
model = load_model('ann_model.h5')

st.title("Insurance Charge Predictor (ANN Model)")
st.write("Enter the details below to predict the insurance charges.")

# Input fields
age = st.slider("Age", 18, 100, 30)
sex = st.selectbox("Sex", ["female", "male"])
bmi = st.slider("BMI", 15.0, 50.0, 25.0)
children = st.slider("Number of Children", 0, 5, 1)
smoker = st.selectbox("Smoker", ["no", "yes"]) # 'no' first as it's the base for drop_first
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"]) # 'northeast' first as it's the base for drop_first

if st.button("Predict"):    
    # Create a dictionary for the input features, matching the one-hot encoded structure
    processed_input = {
        'age': age,
        'bmi': bmi,
        'children': children,
        'sex_male': 1 if sex == 'male' else 0,
        'smoker_yes': 1 if smoker == 'yes' else 0,
        'region_northwest': 1 if region == 'northwest' else 0,
        'region_southeast': 1 if region == 'southeast' else 0,
        'region_southwest': 1 if region == 'southwest' else 0
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([processed_input])

    # Scale the input data using the loaded scaler
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)[0][0]

    st.success(f"Predicted Insurance Charge: ${prediction:,.2f}")
