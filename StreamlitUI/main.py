import os
import pickle
import streamlit as st
import pandas as pd
from pylint.lint import Run


# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the absolute path to the model file
model_path = os.path.join(script_dir, 'rf_model.pkl')

with open(model_path, 'rb') as file:
    model = pickle.load(file)


def predict_temperature(passed_year, passed_month, passed_day, passed_prcp, passed_snwd):
    """
        This function predicts the temperature based on the given parameters.
        Parameters:
        passed_year (float): The year
        passed_month (float): The month
        passed_day (float): The day
        passed_prcp (float): The precipitation
        passed_snwd (float): The snow depth

        Returns:
        float: The predicted average temperature for the provided day
        """
    passed_year, passed_month, passed_day = float(
        passed_year), float(passed_month), float(passed_day)

    passed_prcp = float(passed_prcp) if passed_prcp != "" else None
    passed_snwd = float(passed_snwd) if passed_snwd != "" else None

    latitude, longitude = 52.166, 20.967

    input_data = pd.DataFrame(
        [[latitude, longitude, 110.3, passed_year, passed_month,
            passed_day, passed_prcp, passed_snwd]],
        columns=['LATITUDE', 'LONGITUDE', 'ELEVATION',
                 'Year', 'Month', 'Day', 'PRCP', 'SNWD']
    )

    predicted_temp = model.predict(input_data)
    return predicted_temp[0]


st.title("Temperature Prediction App")

year = st.number_input("Enter Year", min_value=1960,
                       max_value=2025, value=2022)
month = st.number_input("Enter Month", min_value=1, max_value=12, value=1)
day = st.number_input("Enter Day", min_value=1, max_value=31, value=1)
prcp = st.number_input("Enter Precipitation (PRCP)",
                       min_value=0.0, max_value=100.0, value=0.0)
snwd = st.number_input("Enter Snow Depth (SNWD)",
                       min_value=0.0, max_value=100.0, value=0.0)

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


scriptName = "main.py"
# Pylint results
results = Run([scriptName])
print(results.linter.stats.global_note)
