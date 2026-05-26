import os
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ---- Fix File Path ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Load Model ----
model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))

# ---- Load Dataset ----
df = pd.read_csv(os.path.join(BASE_DIR, "car data.csv"))

le = LabelEncoder()
le.fit(df['Car_Name'])
car_names = sorted(df['Car_Name'].unique().tolist())

# ---- Page Config ----
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")
st.title("🚗 Car Price Prediction App")

# ---- Input Fields ----
Car_Name      = st.selectbox("Car Name", car_names)
Present_Price = st.number_input("Present Price (Lakhs ₹)", min_value=0.0, max_value=100.0, value=5.59)
Driven_kms    = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=27000)
Year          = st.number_input("Year of Purchase", min_value=1990, max_value=2024, value=2014)
Fuel_Type     = st.selectbox("Fuel Type", ["CNG", "Diesel", "Petrol"])
Selling_type  = st.selectbox("Seller Type", ["Dealer", "Individual"])
Transmission  = st.selectbox("Transmission", ["Automatic", "Manual"])
Owner         = st.selectbox("Previous Owners", [0, 1, 2, 3])

# ---- Encode ----
fuel_map         = {"CNG": 0, "Diesel": 1, "Petrol": 2}
selling_map      = {"Dealer": 0, "Individual": 1}
transmission_map = {"Automatic": 0, "Manual": 1}

car_encoded  = int(le.transform([Car_Name])[0])
fuel_encoded = fuel_map[Fuel_Type]
Car_Age      = 2024 - int(Year)

features = np.array([[
    car_encoded, float(Present_Price), float(Driven_kms),
    fuel_encoded, selling_map[Selling_type],
    transmission_map[Transmission], Owner,
    fuel_encoded, Car_Age
]])

# ---- Predict ----
if st.button("🔍 Predict Selling Price"):
    prediction = model.predict(features)[0]
    if prediction < 0:
        st.error("⚠️ Could not estimate. Please check inputs.")
    else:
        st.success(f"💰 Estimated Selling Price: ₹ {prediction:.2f} Lakhs")