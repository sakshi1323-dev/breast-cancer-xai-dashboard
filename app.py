import streamlit as st
import joblib
import pandas as pd
import shap
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# ---------------------------
# ✅ Load model & features
# ---------------------------
calibrated_model = joblib.load("best_model.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("🩺 Breast Cancer Prediction Dashboard")

# ---------------------------
# ✅ Sidebar: User inputs
# ---------------------------
st.sidebar.header("Input Features")
user_input = {}
for feature in feature_names:
    user_input[feature] = st.sidebar.number_input(feature, value=0.0)

input_df = pd.DataFrame([user_input])

# ---------------------------
# ✅ Predict button
# ---------------------------
if st.sidebar.button("Predict"):
    # Prediction
    risk_prob = calibrated_model.predict_proba(input_df)[:, 1][0]
    risk_score = round(risk_prob * 100, 2)

    st.subheader("Prediction")
    prediction = "Malignant" if risk_prob > 0.5 else "Benign"
    st.write(f"Predicted Class: **{prediction}**")
    st.write(f"Calibrated Risk Score (0–100): **{risk_score}**")

    # ---------------------------
    # ✅ SHAP Explanation
    # ---------------------------
    st.subheader(" SHAP Feature Importance (Local Explanation)")

    # Theoretical explanation
    st.markdown("""
    **What is SHAP?**  
    SHAP (SHapley Additive exPlanations) is based on cooperative game theory.  
    It assigns each feature a *contribution value* showing how much it pushed the prediction towards **Malignant** or **Benign**.  
    - Positive values → push prediction towards **Malignant**  
    - Negative values → push prediction towards **Benign**  
    """)

    # If model is CalibratedClassifierCV, unwrap estimator
    if isinstance(calibrated_model, CalibratedClassifierCV):
        base_model = calibrated_model.estimator
    else:
        base_model = calibrated_model

    def predict_calibrated(x):
        return calibrated_model.predict_proba(x)

    background = shap.sample(pd.DataFrame([user_input]), 1, random_state=0)

    explainer = shap.KernelExplainer(predict_calibrated, background)
    shap_values = explainer.shap_values(input_df)

    if isinstance(shap_values, list):
        values = shap_values[1][0]  # Malignant class
        base_value = explainer.expected_value[1]
    elif len(shap_values.shape) == 3:
        values = shap_values[0, :, 1]
        base_value = explainer.expected_value[1]
    else:
        values = shap_values[0]
        base_value = explainer.expected_value

    explanation = shap.Explanation(
        values=values,
        base_values=base_value,
        data=input_df.iloc[0].values,
        feature_names=feature_names
    )

    # Graphical explanation
    fig, ax = plt.subplots()
    shap.plots.waterfall(explanation, show=False)
    st.pyplot(fig, bbox_inches="tight")
