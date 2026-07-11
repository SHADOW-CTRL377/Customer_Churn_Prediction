import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf

st.title("Customer Churn Prediction")
CS= st.number_input("Credit Score")
Age= st.number_input("Age")
Tenure= st.number_input("Tenure")
Balance= st.number_input("Balance")
NumOfProducts= st.number_input("Number of Products")
geography= st.selectbox("Geography",("France","Germany","Spain"))
gender= st.selectbox("Gender",("Male","Female")) 
Has_credit_card= st.selectbox("Has Credit Card",("Yes","No"))
Is_active_member= st.selectbox("Is Active Member",("Yes","No"))
Estimated_salary= st.number_input("Estimated Salary")
model = tf.keras.models.load_model("churn_class.h5")
if st.button("Predict"):
  if st.write(model.predict(np.array([[CS,Age,Tenure,Balance,NumOfProducts,geography,gender,Has_credit_card,Is_active_member,Estimated_salary]])))>0.5:
  st.write("Customer will churn")
else:
  st.write("Customer will not churn")
