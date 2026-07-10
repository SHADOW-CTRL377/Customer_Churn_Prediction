import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

@st.cache_resource
def load_resources():
    model = None
    encoder = None
    scaler = None

    # Load model
    try:
        model = tf.keras.models.load_model("customer_churn_model.h5")
    except Exception as e:
        st.error(f"Failed to load model: {e}")

    # Load encoder (optional)
    try:
        with open("encoder.pkl", "rb") as f:
            encoder = pickle.load(f)
    except FileNotFoundError:
        encoder = None
    except Exception as e:
        st.warning(f"Could not load encoder.pkl: {e}")
        encoder = None

    # Load scaler (optional)
    try:
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        scaler = None
    except Exception as e:
        st.warning(f"Could not load scaler.pkl: {e}")
        scaler = None

    return model, encoder, scaler

model, encoder, scaler = load_resources()

st.title("Customer Churn Prediction")

col1, col2 = st.columns(2)
with col1:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
    age = st.number_input("Age", min_value=18, max_value=100, value=40)
    tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=5)

with col2:
    balance = st.number_input("Balance ($)", min_value=0.0, value=50000.0, step=1000.0, format="%.2f")
    num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=0)
    has_cr_card = st.selectbox("Has Credit Card?", ["No", "Yes"])
    is_active_member = st.selectbox("Is Active Member?", ["No", "Yes"])

has_cr_card_val = 1 if has_cr_card == "Yes" else 0
is_active_member_val = 1 if is_active_member == "Yes" else 0

if st.button("Predict Churn"):
    if model is None:
        st.error("Model is not loaded. Please make sure 'customer_churn_model.h5' is present.")
    else:
        # Build feature array in expected order. Adjust order if your model expects something else.
        features = np.array([[credit_score, age, tenure, balance, num_of_products, has_cr_card_val, is_active_member_val]],
                            dtype=np.float32)

        # Apply scaler if available
        try:
            if scaler is not None:
                features = scaler.transform(features)
        except Exception as e:
            st.warning(f"Scaler transform failed: {e}. Using raw features for prediction.")

        # If encoder is required for other categorical fields, apply it here.
        # (For the current two binary fields we've already converted to 0/1.)

        # Predict
        try:
            prediction = model.predict(features, verbose=0)
            # Try to extract probability from different possible shapes
            if prediction.ndim == 2 and prediction.shape[1] == 1:
                prediction_prob = float(prediction[0][0])
            elif prediction.ndim == 2:
                # e.g., softmax with two outputs; assume second column is churn prob
                prediction_prob = float(prediction[0][1])
            else:
                prediction_prob = float(prediction[0])
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            prediction_prob = None

        if prediction_prob is not None:
            st.markdown("---")
            st.subheader("Prediction Result")
            st.write(f"**Churn Probability:** {prediction_prob:.2%}")

            if prediction_prob >= 0.5:
                st.error("⚠️ This customer is **likely to churn**.")
            else:
                st.success("✅ This customer is **likely to stay**.")
