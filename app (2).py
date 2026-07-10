import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model("customer_churn_model.h5")

    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, encoder, scaler

model, encoder, scaler = load_resources()
with col1:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
    age = st.number_input("Age", min_value=18, max_value=100, value=40)
    tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=5)

with col2:
    balance = st.number_input("Balance ($)", min_value=0.0, value=50000.0, step=1000.0)
    num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=0)
    has_cr_card = st.selectbox("Has Credit Card?", ["No", "Yes"])
    is_active_member = st.selectbox("Is Active Member?", ["No", "Yes"])

has_cr_card_val = 1 if has_cr_card == "Yes" else 0
is_active_member_val = 1 if is_active_member == "Yes" else 0

if st.button("Predict Churn", type="primary"):
    features = np.array([[credit_score, age, tenure, balance, num_of_products, has_cr_card_val, is_active_member_val]])
    prediction_prob = model.predict(features)[0][0]

    st.markdown("---")
    st.subheader("Prediction Result")
    st.write(f"**Churn Probability:** {prediction_prob:.2%}")

    if prediction_prob >= 0.5:
        st.error("⚠️ This customer is **likely to churn**.")
    else:
        st.success("✅ This customer is **likely to stay**.")
