import pandas as pd
import joblib
import numpy as np
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin


# ── FrequencyEncoder (must match training) ───────────────────────────────────
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.maps = {}
        self.cols = cols

    def fit(self, X, y=None):
        for col in self.cols:
            self.maps[col] = X[col].value_counts(normalize=True)
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].map(self.maps[col]).fillna(0)
        return X


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Used Car Price Predictor", layout="centered")
st.title("🚗 Used Car Price Predictor")
st.caption("XGBoost model · R² 0.869 on test set")


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(r"E:\AI_Projects\Machine_Learning\USED_CARS_PRICE_PREDECTION\notebooks\ML_Models\best_xgboost.pk1")

model = load_model()


# ── Inputs ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    brand        = st.text_input("Brand", "Ford")
    model_name   = st.text_input("Model", "F-150 XLT")
    model_year   = st.number_input("Model Year", 1990, 2025, 2018)
    milage       = st.number_input("Mileage (mi)", 0, 500000, 45000)
    fuel_type    = st.selectbox("Fuel Type", ["Gasoline", "Diesel", "Electric", "Hybrid", "E85 Flex Fuel", "Plug-In Hybrid"])
    accident     = st.selectbox("Accident Occurred", ["No", "Yes"])

with col2:
    transmission      = st.text_input("Transmission", "8-Speed Automatic")
    ext_col           = st.text_input("Exterior Color", "White")
    int_col           = st.text_input("Interior Color", "Black")
    clean_title       = st.selectbox("Clean Title", ["Yes", "No", "unknown"])
    engine_capacity   = st.number_input("Engine Capacity (L)", 0.0, 10.0, 2.5, step=0.1)
    horse_power       = st.number_input("Horse Power (HP)", 0, 2000, 200)


# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("⚡ Predict Price", use_container_width=True, type="primary"):

    # Column names MUST match what the pipeline was trained on
    input_df = pd.DataFrame([{
        "brand":                    brand,
        "model":                    model_name,
        "model_year":               int(model_year),
        "milage":                   float(milage),        # ← milage (not mileage)
        "fuel_type":                fuel_type,
        "transmission":             transmission,
        "ext_col":                  ext_col,
        "int_col":                  int_col,
        "clean_title":              clean_title,
        "Engine_Capacity":          float(engine_capacity),   # ← capital E and C
        "Horse_Power":              float(horse_power),        # ← capital H and P
        "Engine_Capacity_Missing":  0,                         # ← required flag col
        "Accident_occured":         accident,                  # ← capital A, lowercase o
    }])

    try:
        predicted_price = model.predict(input_df)[0]
        st.success(f"### Estimated Price: ${predicted_price:,.0f}")
        st.write(f"**{brand} {model_name}** · {int(model_year)} · {int(milage):,} mi")
    except Exception as e:
        st.error(f"Prediction error: {e}")
