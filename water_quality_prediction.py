import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="Water Quality Prediction", layout="centered")

st.title("💧 Water Quality Prediction using XGBoost")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset here", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ----------------------------
    # Model Training
    # ----------------------------
    X = df.drop("Potability", axis=1)
    y = df["Potability"]

    # Handle NaN values
    X = X.fillna(X.mean())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train XGBoost model
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"### ✅ Model Accuracy: {acc:.4f}")

    # ----------------------------
    # User Input for Prediction
    # ----------------------------
    st.subheader("🔮 Test with Your Own Values")

    input_data = {}
    for col in X.columns:
        val = st.number_input(f"Enter {col}", value=float(X[col].mean()))
        input_data[col] = val

    if st.button("Predict Potabilty"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]

        if prediction == 1:
            st.success("✅ Safe for Drinking (Potable - Good Quality)")
        else:
            st.error("❌ Not Safe for Drinking (Not Potable - Poor Quality)")
