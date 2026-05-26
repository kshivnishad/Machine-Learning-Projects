import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

model = pickle.load(open("model.pkl", "rb"))

df = pd.read_csv("car data.csv")
le = LabelEncoder()
le.fit(df['Car_Name'])
car_names = sorted(df['Car_Name'].unique().tolist())

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")
st.title("🚗 Car Price Prediction App")

Car_Name      = st.selectbox("Car Name", car_names)
Present_Price = st.number_input("Present Price (Lakhs ₹)", min_value=0.0, max_value=100.0, value=5.59)
Driven_kms    = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=27000)
Year          = st.number_input("Year of Purchase", min_value=1990, max_value=2024, value=2014)
Fuel_Type     = st.selectbox("Fuel Type", ["CNG", "Diesel", "Petrol"])
Selling_type  = st.selectbox("Seller Type", ["Dealer", "Individual"])
Transmission  = st.selectbox("Transmission", ["Automatic", "Manual"])
Owner         = st.selectbox("Previous Owners", [0, 1, 2, 3])

fuel_map         = {"CNG": 0, "Diesel": 1, "Petrol": 2}
selling_map      = {"Dealer": 0, "Individual": 1}
transmission_map = {"Automatic": 0, "Manual": 1}

car_encoded  = int(le.transform([Car_Name])[0])
fuel_encoded = fuel_map[Fuel_Type]
Car_Ag       = 2024 - int(Year)

features = np.array([[
    car_encoded,                    # 1. Car_Name
    float(Present_Price),           # 2. Present_Price
    float(Driven_kms),              # 3. Driven_kms
    fuel_encoded,                   # 4. Fuel_Type
    selling_map[Selling_type],      # 5. Selling_type
    transmission_map[Transmission], # 6. Transmission
    Owner,                          # 7. Owner
    fuel_encoded,                   # 8. Fuel_Type =
    Car_Ag                          # 9. Car_Ag
]])

st.write(f"✅ Features shape: {features.shape}")  # should show (1, 9)

if st.button("🔍 Predict Selling Price"):
    prediction = model.predict(features)[0]
    if prediction < 0:
        st.error("⚠️ Could not estimate. Please check inputs.")
    else:
        st.success(f"💰 Estimated Selling Price: ₹ {prediction:.2f} Lakhs")