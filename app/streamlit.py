import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model, scaler, and selected features
@st.cache_resource
def load_resources():
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("model/selected_features.pkl", "rb") as f:
        selected_features = pickle.load(f)
    return model, scaler, selected_features

model, scaler, selected_features = load_resources()

st.title("Credit Risk Prediction App")
st.subheader("Enter Applicant Details")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
duration = st.number_input("Credit Duration (months)", min_value=1, max_value=100, value=12)
credit_amount = st.number_input("Credit Amount", min_value=100, max_value=100000, value=2000)

sex = st.selectbox("Sex", options=["male", "female"])
job = st.selectbox("Job (0=unskilled, 3=highly skilled)", options=[0, 1, 2, 3])
housing = st.selectbox("Housing", options=["own", "rent", "free"])
saving_accounts = st.selectbox("Saving Accounts", options=["little", "moderate", "quite rich", "rich"])
checking_account = st.selectbox("Checking Account", options=["little", "moderate", "rich"])
purpose = st.selectbox("Purpose", options=[
    "radio/TV", "education", "furniture/equipment", "new car",
    "used car", "business", "domestic appliance", "repairs", "vacation/others"
])

# Encoding
sex_encoded = 0 if sex == "male" else 1
housing_map = {"own": 0, "rent": 1, "free": 2}
saving_map = {"little": 0, "moderate": 1, "quite rich": 2, "rich": 3}
checking_map = {"little": 0, "moderate": 1, "rich": 2}
purpose_map = {
    "radio/TV": 0, "education": 1, "furniture/equipment": 2,
    "new car": 3, "used car": 4, "business": 5, "domestic appliance": 6,
    "repairs": 7, "vacation/others": 8
}

housing_encoded = housing_map[housing]
saving_encoded = saving_map[saving_accounts]
checking_encoded = checking_map[checking_account]
purpose_encoded = purpose_map[purpose]

# Create full feature dictionary
all_features = {
    "Age": age,
    "Duration": duration,
    "Credit amount": credit_amount,
    "Sex": sex_encoded,
    "Job": job,
    "Housing": housing_encoded,
    "Saving accounts": saving_encoded,
    "Checking account": checking_encoded,
    "Purpose": purpose_encoded
}

# Only select features used during training
input_data = pd.DataFrame([{k: all_features[k] for k in selected_features}])

# Prediction
if st.button("Predict"):
    try:
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        prediction_label = "Good Credit Risk" if prediction == 1 else "Bad Credit Risk"
        st.success(f"Prediction: {prediction_label}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")