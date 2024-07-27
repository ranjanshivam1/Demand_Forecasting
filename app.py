import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pycaret.time_series import *
from PIL import Image
import requests
from streamlit_lottie import st_lottie
import time

# Function to load Lottie animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Function to calculate and display metrics
def calculate_metrics(model_results, test_data):
    metrics = {}
    for model_name, model in model_results.items():
        y_true = test_data.values
        y_pred = model.predict()  # Make sure to adjust this based on how your model provides predictions
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mape = np.mean(np.abs((y_true - y_pred) / y_true))
        smape = np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
        r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
        tt = time.time() - start_time  # Example, adjust based on your timing needs
        metrics[model_name] = [mae, rmse, mape, smape, r2, tt]
    
    return pd.DataFrame(metrics, index=['MAE', 'RMSE', 'MAPE', 'SMAPE', 'R2', 'TT (Sec)']).T

# Add Lottie animation for processing
def show_processing_animation():
    lottie_url = "https://assets4.lottiefiles.com/packages/lf20_4sqkv2wq.json"  # URL for the animation
    lottie_data = load_lottieurl(lottie_url)
    if lottie_data:
        st_lottie(lottie_data, speed=1, loop=True, quality="high", height=300, width=300)

# Add the title and side images
st.title("Web Based Demand Forecasting Application")
st.caption("Upload your CSV file for demand forecasting")
lottie_hello = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_49rdyysj.json")
st_lottie(lottie_hello, speed=1, reverse=True, loop=True, quality="high", height=250, width=1200, key=None)
st.sidebar.title("Demand Forecasting Application")
logo = Image.open('image3.jpg')
st.sidebar.image(logo, width=300)
st.sidebar.caption("Library Used - PyCaret")
st.sidebar.caption("Designed by: Your Name")

# User input for number of periods
fh = st.number_input("Enter the number of periods to forecast", min_value=1, max_value=365, value=10)

# Upload file
data_file = st.file_uploader("Upload a CSV file", type=['csv'])
if data_file:
    data_file = pd.read_csv(data_file)
    data_file['Date'] = pd.to_datetime(data_file['Date'])
    data_file.set_index('Date', inplace=True)
    train_size = int(len(data_file) * 0.8)
    train_data = data_file.iloc[:train_size]
    test_data = data_file.iloc[train_size:]

    # Show processing animation
    with st.spinner('Processing...'):
        show_processing_animation()
        # Train models
        start_time = time.time()
        s = setup(train_data, fh=fh, fold=3)
        models_to_compare = ['ets', 'arima', 'prophet', 'naive', 'exp_smooth', 'polytrend', 'croston', 'gbr_cds_dt']
        best_model = compare_models(include=models_to_compare)
        final_best = finalize_model(best_model)
        future_forecast = predict_model(final_best, fh=fh)
        
        # Calculate metrics
        model_results = {}  # Replace with actual model results if needed
        metrics_df = calculate_metrics(model_results, test_data)
        
        # Display metrics
        st.write("Model Metrics:")
        st.dataframe(metrics_df)
        
        # Plot and display forecast
        plt.figure(figsize=(12, 6))
        plt.plot(test_data.index, test_data.values, label='Actual', color='blue')
        forecast_index = pd.date_range(start=test_data.index[-1] + pd.DateOffset(days=1), periods=len(future_forecast), freq=test_data.index.freq)
        future_forecast.index = forecast_index
        plt.plot(future_forecast.index, future_forecast[future_forecast.columns[0]], label='Forecast', color='red')
        plt.title('Actual vs Forecasted Values')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig('/content/forecast_vs_actual.png')
        st.image('/content/forecast_vs_actual.png')
        
        # Display forecast table
        forecast_file = '/content/future_forecast.xlsx'
        future_forecast.to_excel(forecast_file, index=True)
        st.write("Forecast values have been saved.")
        st.download_button("Download Forecast Data", data=open(forecast_file, 'rb'), file_name='future_forecast.xlsx')

# Add button for submission after entering periods
if st.button("Submit"):
    st.write("Processing your request...")
    # Re-trigger processing logic here if needed
