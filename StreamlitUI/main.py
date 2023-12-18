import os
import streamlit as st
import pandas as pd
import pickle


# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the absolute path to the model file
model_path = os.path.join(script_dir, 'oldModel/knn_model.pkl')

with open(model_path, 'rb') as file:
    model = pickle.load(file)


def predict_temperature(year, month, day, prcp, snwd):
    year, month, day = float(year), float(month), float(day)

    prcp = float(prcp) if prcp != "" else None
    snwd = float(snwd) if snwd != "" else None

    latitude, longitude = 52.166, 20.967

    input_data = pd.DataFrame([[latitude, longitude, 110.3, year, month, day, prcp, snwd]],
                              columns=['LATITUDE', 'LONGITUDE', 'ELEVATION', 'Year', 'Month', 'Day', 'PRCP', 'SNWD'])

    prediction = model.predict(input_data)
    return prediction[0]


st.title("Temperature Prediction App")

year = st.number_input("Enter Year", min_value=1960, max_value=2025, value=2022)
month = st.number_input("Enter Month", min_value=1, max_value=12, value=1)
day = st.number_input("Enter Day", min_value=1, max_value=31, value=1)
prcp = st.number_input("Enter Precipitation (PRCP)", min_value=0.0, max_value=100.0, value=0.0)
snwd = st.number_input("Enter Snow Depth (SNWD)", min_value=0.0, max_value=100.0, value=0.0)

if st.button("Predict Temperature"):
    prediction = predict_temperature(year, month, day, prcp, snwd)
    st.success(f"The predicted temperature is: {prediction:.2f} °C")

    # Check if the prediction is positive or negative
    if prediction < 0:
        st.warning("It's cold! Let it snow!")
        st.snow()
    elif prediction > 8:
        st.success("It's warm! Enjoy the balloons!")
        st.balloons()
    else:
        st.info("The temperature is just right!")
