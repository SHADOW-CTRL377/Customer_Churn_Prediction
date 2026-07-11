import streamlit as st
import numpy as np
import tensorflow as tf

st.title("Customer Churn Prediction")

CS = st.number_input("Credit Score")
Age = st.number_input("Age")
Tenure = st.number_input("Tenure")
Balance = st.number_input("Balance")
NumOfProducts = st.number_input("Number of Products")

geography = st.selectbox("Geography", ("France", "Germany", "Spain"))
gender = st.selectbox("Gender", ("Male", "Female"))
Has_credit_card = st.selectbox("Has Credit Card", ("Yes", "No"))
Is_active_member = st.selectbox("Is Active Member", ("Yes", "No"))

Estimated_salary = st.number_input("Estimated Salary")

# Convert text to numbers
geo_map = {"France": 0, "Germany": 1, "Spain": 2}
gender_map = {"Male": 1, "Female": 0}
card_map = {"Yes": 1, "No": 0}
active_map = {"Yes": 1, "No": 0}

model = tf.keras.models.load_model("churn_class.h5")

if st.button("Predict"):

    input_data = np.array([[CS,
                            Age,
                            Tenure,
                            Balance,
                            NumOfProducts,
                            geo_map[geography],
                            gender_map[gender],
                            card_map[Has_credit_card],
                            active_map[Is_active_member],
                            Estimated_salary]])

    prediction = model.predict(input_data)

    st.write("Prediction Value:", prediction[0][0])

    if prediction[0][0] > 0.5:
        st.success("Customer will churn")
    else:
        st.success("Customer will not churn")
