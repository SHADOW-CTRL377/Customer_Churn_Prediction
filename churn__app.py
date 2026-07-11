import streamlit as st
import numpy as np
import tensorflow as tf

st.title("Customer Churn Prediction")

CS = st.number_input("Credit Score", min_value=0, max_value=1000, value=600)
Age = st.number_input("Age", min_value=18, max_value=100, value=30)
Tenure = st.number_input("Tenure", min_value=0, max_value=50, value=5)
Balance = st.number_input("Balance", min_value=0.0, value=10000.0)
NumOfProducts = st.number_input("Number of Products", min_value=1, max_value=4, value=1)

geography = st.selectbox("Geography", ("France", "Germany", "Spain"))
gender = st.selectbox("Gender", ("Male", "Female"))
Has_credit_card = st.selectbox("Has Credit Card", ("Yes", "No"))
Is_active_member = st.selectbox("Is Active Member", ("Yes", "No"))

Estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

# Mappings
gender_map = {"Male": 1, "Female": 0}
card_map = {"Yes": 1, "No": 0}
active_map = {"Yes": 1, "No": 0}

# One-hot encoding for Geography (3 features)
geo_one_hot = {
    "France": [1, 0, 0],
    "Germany": [0, 1, 0],
    "Spain": [0, 0, 1]
}

# Load model with caching
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("churn_class.h5")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

if st.button("Predict"):
    try:
        # Create input array with 12 features: [CS, Age, Tenure, Balance, NumOfProducts, 
        # Gender, HasCreditCard, IsActiveMember, EstimatedSalary, Geo_France, Geo_Germany, Geo_Spain]
        input_data = np.array([[
            CS,
            Age,
            Tenure,
            Balance,
            NumOfProducts,
            gender_map[gender],
            card_map[Has_credit_card],
            active_map[Is_active_member],
            Estimated_salary,
            geo_one_hot[geography][0],  # France
            geo_one_hot[geography][1],  # Germany
            geo_one_hot[geography][2]   # Spain
        ]], dtype=np.float32)

        prediction = model.predict(input_data, verbose=0)

        st.write(f"Prediction Score: {prediction[0][0]:.4f}")

        if prediction[0][0] > 0.5:
            st.error("⚠️ Customer will likely **CHURN**")
        else:
            st.success("✅ Customer will **NOT CHURN**")
    
    except Exception as e:
        st.error(f"Prediction error: {e}")
