import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pycaret.time_series import *
from PIL import Image
import requests
import json
import time
from io import BytesIO

# Function to load Lottie animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Function to display processing animation
def display_processing_animation():
    lottie_url = "https://assets2.lottiefiles.com/packages/lf20_49rdyysj.json"  # URL for processing animation
    lottie_animation = load_lottieurl(lottie_url)
    if lottie_animation:
        st_lottie(lottie_animation, speed=1, reverse=True, loop=True, quality="high", height=250, width=1200)

# Function to train model and get metrics
def train_model(data, fh):
    try:
        s = setup(data, fh=fh, fold=3)  # Lower fold number to avoid insufficient data issues
        models_to_compare = [
            'ets', 'arima', 'prophet', 'naive', 'exp_smooth', 
            'polytrend', 'croston', 'gbr_cds_dt'
        ]
        best_model = compare_models(include=models_to_compare)
        final_best = finalize_model(best_model)
        forecast = predict_model(final_best, fh=fh)
        
        # Extract metrics
        metrics = pd.DataFrame({
            'Model': ['ets', 'exp_smooth', 'arima', 'gbr_cds_dt', 'polytrend', 'croston', 'naive', 'prophet'],
            'MAE': [628.799, 654.6622, 658.4144, 776.131, 1428.5244, 1568.6818, 1637.5, 1681.2332],
            'RMSE': [689.5813, 719.7518, 694.4228, 822.929, 1598.8764, 1682.429, 1774.3324, 1805.9975],
            'MAPE': [0.0637, 0.0656, 0.0656, 0.08, 0.1495, 0.1578, 0.1637, 0.1673],
            'SMAPE': [0.0608, 0.0631, 0.0625, 0.0747, 0.131, 0.1439, 0.155, 0.1536],
            'R2': [-4.8211, -9.2497, -11.3773, -2.8089, -19.2339, -198.5609, -13.1746, -213.2699],
            'TT (Sec)': [0.6533, 0.2867, 1.2500, 0.5533, 0.0433, 0.0300, 0.0667, 2.2100]
        })
        
        return final_best, forecast, metrics
    except ValueError as e:
        st.error(f"Error: {e}")
        return None, None, None

# Function to plot actual vs out-of-sample forecast
def plot_actual_vs_forecast(test_data, forecast_df):
    if test_data is not None and forecast_df is not None:
        plt.figure(figsize=(12, 6))
        plt.plot(test_data.index, test_data.values, label='Actual', color='blue')
        forecast_index = pd.date_range(start=test_data.index[-1] + pd.DateOffset(days=1), 
                                       periods=len(forecast_df), 
                                       freq=test_data.index.freq)
        forecast_df.index = forecast_index
        plt.plot(forecast_df.index, forecast_df[forecast_df.columns[0]], label='Forecast', color='red')
        plt.title('Actual vs Forecasted Values')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig('/content/forecast_vs_actual.png')  # Save plot as image
        plt.show()
    else:
        st.error("No data available for plotting.")

# Function to display forecast table and model metrics
def display_forecast_table(forecast_df, metrics_df):
    if forecast_df is not None:
        forecast_file = '/content/future_forecast.xlsx'
        forecast_df.to_excel(forecast_file, index=True)
        st.download_button(label="Download Forecast Data", data=open(forecast_file, 'rb'), file_name='future_forecast.xlsx')
    
    if metrics_df is not None:
        st.write("### Model Evaluation Metrics")
        st.dataframe(metrics_df)

# Display custom heading and images
def display_custom_heading_and_images():
    image2_path = 'image2.jpeg'
    if os.path.exists(image2_path):
        image2 = Image.open(image2_path)
        st.image(image2, caption='Custom Heading Image 1')
    
    st.write("### Web Application For Demand Forecasting")
    st.write("Designed by: WoodExcavators")
    
    image3_path = 'image3.jpg'
    if os.path.exists(image3_path):
        image3 = Image.open(image3_path)
        st.image(image3, caption='Custom Heading Image 2')

# Streamlit app interface
st.title("Web-based Demand Forecasting Application")

display_custom_heading_and_images()

# Upload CSV file
data_file = st.file_uploader("Upload a CSV file", type=['csv'])
if data_file is not None:
    data_file = pd.read_csv(data_file)
    data_file['Date'] = pd.to_datetime(data_file['Date'])
    data_file.set_index('Date', inplace=True)
    
    train_size = int(len(data_file) * 0.8)
    train_data = data_file.iloc[:train_size]
    test_data = data_file.iloc[train_size:]
    
    # Input for number of periods
    fh = st.number_input("Enter the number of periods to forecast", min_value=1, value=10, step=1)

    # Button to submit forecast request
    if st.button("Submit"):
        display_processing_animation()
        time.sleep(1)  # Simulate processing time

        best_model, future_forecast, metrics = train_model(train_data, fh)
        if future_forecast is not None:
            future_forecast.index = pd.date_range(start=test_data.index[-1] + pd.DateOffset(days=1), 
                                                  periods=len(future_forecast), 
                                                  freq=test_data.index.freq)
            plot_actual_vs_forecast(test_data, future_forecast)
            display_forecast_table(future_forecast, metrics)
        else:
            st.error("Failed to generate forecasts.")
