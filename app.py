import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction")
st.write("Enter the customer details below to predict whether the customer is likely to churn.")

st.markdown("---")

# User Inputs
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)

geography = st.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

age = st.number_input("Age", min_value=18, max_value=100, value=35)

tenure = st.slider("Tenure (Years)", 0, 10, 5)

balance = st.number_input("Balance", min_value=0.0, value=50000.0)

num_products = st.selectbox(
    "Number of Products",
    [1, 2, 3, 4]
)

has_card = st.selectbox(
    "Has Credit Card?",
    ["Yes", "No"]
)

active_member = st.selectbox(
    "Is Active Member?",
    ["Yes", "No"]
)

salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=100000.0
)

# Encoding
gender = 1 if gender == "Male" else 0

has_card = 1 if has_card == "Yes" else 0

active_member = 1 if active_member == "Yes" else 0

geo_germany = 0
geo_spain = 0

if geography == "Germany":
    geo_germany = 1

elif geography == "Spain":
    geo_spain = 1

# Prediction Button
if st.button("Predict Churn"):

    features = np.array([[
        credit_score,
        gender,
        age,
        tenure,
        balance,
        num_products,
        has_card,
        active_member,
        salary,
        geo_germany,
        geo_spain
    ]])

    scaled_features = scaler.transform(features)

    prediction = model.predict(scaled_features)[0]

    probability = model.predict_proba(scaled_features)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.error("⚠️ Customer is likely to CHURN.")
    else:
        st.success("✅ Customer is NOT likely to churn.")

    st.subheader("Prediction Probability")

    st.progress(float(probability))

    st.write(f"**Churn Probability:** {probability:.2%}")