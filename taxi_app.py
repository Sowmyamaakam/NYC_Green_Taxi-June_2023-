import streamlit as st
import numpy as np
import joblib
import gdown
import os
from sklearn.preprocessing import StandardScaler

# Function to download file from Google Drive
def download_model(file_id, output_path='best_model_rf_top10.pkl'):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_path, quiet=False)

# Download the model file (run this only if the file doesn't exist)
file_id = '1c_cLhpFgxoRgowgrfx90y49o-JwFMjAv'  # File ID from your Google Drive link
if not os.path.exists('best_model_rf_top10.pkl'):
    download_model(file_id)

# Load the pre-trained model and scaler
scaler = joblib.load('num_scaler.pkl')
model = joblib.load('best_model_rf_top10.pkl')
important_features = joblib.load('important_features_top10.pkl')

# Title for the app
st.title('NYC Taxi Fare Prediction')

# Description
st.write("""
    This app predicts the taxi fare based on various trip details. 
    Fill in the information below to get an estimated fare.
""")

# Input fields for trip details
trip_distance = st.number_input('Trip Distance (miles)', min_value=0.0, value=1.0, step=0.1)
trip_duration = st.number_input('Trip Duration (minutes)', min_value=0.0, value=10.0, step=1.0)
tip_amount = st.number_input('Tip Amount ($)', min_value=0.0, value=2.0, step=0.1)
rate_code = st.selectbox('Rate Code', [1, 2, 3, 4, 5, 6], index=0)
do_zone = st.number_input('Drop-off Zone Encoding (DOZone)', min_value=0.0, value=0.0, step=0.1)
drop_off_hour = st.slider('Drop-off Hour', 0, 23, 0)
day_of_week = st.selectbox('Day of the Week', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
pu_zone = st.number_input('Pick-up Zone Encoding (PUZone)', min_value=0.0, value=0.0, step=0.1)
passenger_count = st.selectbox('Passenger Count', [1, 2, 3, 4, 5, 6], index=0)
mta_tax = st.number_input('MTA Tax ($)', min_value=0.0, value=0.5, step=0.1)

# Convert the Day of the Week to numeric value
day_of_week_mapping = {
    'Monday': 0,
    'Tuesday': 1,
    'Wednesday': 2,
    'Thursday': 3,
    'Friday': 4,
    'Saturday': 5,
    'Sunday': 6
}

# Prepare the input data for prediction
if st.button('Predict Fare'):
    # Convert the day of the week to numeric
    day_of_week_numeric = day_of_week_mapping[day_of_week]
    
    # Prepare the feature vector based on the inputs
    features = np.array([
        trip_distance,
        trip_duration,
        tip_amount,
        rate_code,
        do_zone,
        drop_off_hour,
        day_of_week_numeric,
        pu_zone,
        passenger_count,
        mta_tax
    ]).reshape(1, -1)

    # Scale the features
    scaled_features = scaler.transform(features)

    # Predict the fare using the model
    predicted_fare = model.predict(scaled_features)

    # Show the predicted fare
    st.write(f"### Predicted Taxi Fare: ${predicted_fare[0]:.2f}")
