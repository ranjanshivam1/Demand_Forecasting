import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pycaret.time_series import *
from PIL import Image
import io
import requests
import os

# Function to load Lottie animation (mocked with a placeholder image)
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Define model training function with specified models
def train_model(data, fh):
    try:
        s = setup(data, fh=fh, fold=3)  # Lower fold number to avoid insufficient data issues
        # List of models to compare
        models_to_compare = [
            'ets', 'arima', 'prophet', 'naive', 'exp_smooth', 
            'polytrend', 'croston', 'gbr_cds_dt'
        ]
        best_model = compare_models(include=models_to_compare)
        final_best = finalize_model(best_model)
        # Generate future forecasts
        future_forecast = predict_model(final_best, fh=fh)
        return final_best, future_forecast
    except ValueError as e:
        st.write(f"Error: {e}")
        return None, None

# Function to plot actual vs out-of-sample forecast
def plot_actual_vs_forecast(test_data, forecast_df):
    if test_data is not None and forecast_df is not None:
        plt.figure(figsize=(12, 6))

        # Plot actual values
        plt.plot(test_data.index, test_data.values, label='Actual', color='blue')

        # Align forecast_df index with the same frequency as test_data
        forecast_index = pd.date_range(start=test_data.index[-1] + pd.DateOffset(days=1), 
                                       periods=len(forecast_df), 
                                       freq=test_data.index.freq)
        forecast_df.index = forecast_index

        # Plot forecast values
        plt.plot(forecast_df.index, forecast_df[forecast_df.columns[0]], label='Forecast', color='red')

        plt.title('Actual vs Forecasted Values')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig('forecast_vs_actual.png')  # Save plot as image
        st.pyplot(plt)
    else:
        st.write("No data available for plotting.")

# Function to get and display future forecast values
def display_forecast_table(forecast_df):
    if forecast_df is not None:
        # Save forecast values to Excel file
        forecast_file = 'future_forecast.xlsx'
        forecast_df.to_excel(forecast_file, index=True)
        st.write(f"Forecast values have been saved to {forecast_file}")

        # Provide download link
        with open(forecast_file, 'rb') as f:
            st.download_button(label='Download Forecast', data=f, file_name=forecast_file)
    else:
        st.write("No forecast data available.")

# Function to parse dates with mixed formats and infer format
def parse_dates(date_series):
    try:
        # Try parsing dates with different formats
        return pd.to_datetime(date_series, infer_datetime_format=True)
    except ValueError:
        raise ValueError("Date format cannot be inferred or parsed.")

# Function to display custom heading and images
def display_custom_heading_and_images():
    # Display image2.jpeg
    image2_path = 'image2.jpeg'  # Ensure the image file is uploaded in the same directory
    if os.path.exists(image2_path):
        image2 = Image.open(image2_path)
        st.image(image2, caption='Web Application For Demand Forecasting')
    else:
        st.write(f"File not found: {image2_path}")

    # Display custom headings
    st.markdown("""
    <h1 style='text-align: center; color: black;'>Web Application For Demand Forecasting</h1>
    <h3 style='text-align: center; color: maroon;'>Designed by: WoodExcavators</h3>
    """, unsafe_allow_html=True)

    # Display image3.jpg
    image3_path = 'image3.jpg'  # Ensure the image file is uploaded in the same directory
    if os.path.exists(image3_path):
        image3 = Image.open(image3_path)
        st.image(image3, caption='Lottie Animation')
    else:
        st.write(f"File not found: {image3_path}")

# Display custom heading and images
display_custom_heading_and_images()

# Instructions for uploading CSV file
st.write("Please upload a CSV file. The CSV file should have the following format:")
st.write("1. The first column should be named 'Date'.")
st.write("2. The target values should be in the next consecutive column.")
st.write("3. Date format can be either mm/dd/yyyy or dd/mm/yyyy.")

# Upload and process the CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data_file = pd.read_csv(uploaded_file)

    # Parse dates and ensure consistency
    data_file['Date'] = parse_dates(data_file['Date'])
    data_file.set_index('Date', inplace=True)

    # Ensure index is consistent and infer frequency
    if data_file.index.freq is None:
        # Try inferring the frequency if it's not set
        inferred_freq = pd.infer_freq(data_file.index)
        if inferred_freq:
            data_file.index = data_file.index.to_period(inferred_freq).to_timestamp()
        else:
            data_file.index = data_file.index.to_period('M').to_timestamp()

    # Split data into training and testing sets
    train_size = int(len(data_file) * 0.8)
    train_data = data_file.iloc[:train_size]
    test_data = data_file.iloc[train_size:]

    # Ask user for number of periods to forecast
    fh = st.number_input("Enter the number of periods to forecast (e.g., 10, 30, 60): ", min_value=1, max_value=100, value=10)

    # Ensure the forecast horizon does not exceed the number of available test periods
    fh = min(fh, len(test_data))

    # Train model and get forecasts
    best_model, future_forecast = train_model(train_data, fh)

    # Convert future_forecast index to match test_data's frequency
    if future_forecast is not None:
        future_forecast.index = pd.date_range(start=test_data.index[-1] + pd.DateOffset(days=1), 
                                              periods=len(future_forecast), 
                                              freq=test_data.index.freq)

    # Plot actual vs forecasted values
    plot_actual_vs_forecast(test_data, future_forecast)

    # Display future forecast values as a table and save to Excel
    display_forecast_table(future_forecast)
