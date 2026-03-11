import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load dataset and train model
df = pd.read_csv("heart.csv")
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

X = df.drop('heart_disease', axis=1)
y = df['heart_disease']
X = pd.get_dummies(X, drop_first=True)

model = LogisticRegression(max_iter=2000)
model.fit(X, y)

# Streamlit GUI
st.title("Heart Disease Prediction App")
st.subheader("Enter Patient Details:")

age = st.number_input("Age", 1, 120, 50)
gender = st.selectbox("Gender", ["Male", "Female"])
hypertension = st.selectbox("Hypertension (0=No,1=Yes)", [0,1])
ever_married = st.selectbox("Ever Married?", ["Yes","No"])
work_type = st.selectbox("Work Type", ["Private","Self-employed","Govt_job","children","Never_worked"])
Residence_type = st.selectbox("Residence Type", ["Urban","Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", value=120.0)
bmi = st.number_input("BMI", value=25.0)
smoking_status = st.selectbox("Smoking Status", ["never smoked","formerly smoked","smokes","Unknown"])
stroke = st.selectbox("Stroke (0=No,1=Yes)", [0,1])

if st.button("Predict Heart Disease"):
    input_data = {
        'id': 0,
        'age': age,
        'gender': gender,
        'hypertension': hypertension,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': Residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status,
        'stroke': stroke
    }
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)
    
    # Add missing columns
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Ensure same column order
    input_df = input_df[X.columns]
    
    # Prediction
    prediction = model.predict(input_df)
    if prediction[0] == 1:
        st.error("Person has Heart Disease Risk")
    else:
        st.success("Person has No Heart Disease Risk")