import pandas as pd
import matplotlib.pyplot as plt
from pycaret.time_series import *
from PIL import Image
import streamlit as st
import requests
import io
import numpy as np

# Function to load Lottie animation (mocked with a placeholder image)
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Function to calculate and return error metrics for each model
def calculate_error_metrics(models, test_data, fh):
    metrics = []
    for model_name, model in models.items():
        # Use the model to make predictions
        predictions = predict_model(model, fh=fh)
        # Calculate error metrics
        actual = test_data.iloc[:len(predictions)]
        mae = np.mean(np.abs(actual - predictions))
        mse = np.mean((actual - predictions) ** 2)
        rmse = np.sqrt(mse)
        metrics.append({
            'Model': model_name,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse
        })
    return pd.DataFrame(metrics)

# Define model training function with specified models
def train_model(data, fh):
    s = setup(data, fh=fh, fold=3)
    models_to_compare = ['ets', 'arima', 'prophet', 'naive', 'exp_smooth', 'polytrend', 'croston', 'gbr_cds_dt']
    best_model = compare_models(include=models_to_compare)
    final_best = finalize_model(best_model)
    future_forecast = predict_model(final_best, fh=fh)
    
    # Get all models and calculate error metrics
    all_models = pull_models()
    error_metrics = calculate_error_metrics(all_models, data, fh)
    
    return final_best, future_forecast, error_metrics

def plot_actual_vs_forecast(test_data, forecast_df):
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
    plt.savefig('forecast_vs_actual.png')
    plt.show()

def display_forecast_table(forecast_df):
    forecast_file = 'future_forecast.xlsx'
    forecast_df.to_excel(forecast_file, index=True)
    st.download_button('Download Forecast Data', data=open(forecast_file, 'rb').read(), file_name=forecast_file)

def parse_dates(date_series):
    return pd.to_datetime(date_series, infer_datetime_format=True)

st.title("Web Application For Demand Forecasting")
st.caption("Use this web-based application to forecast your demands - upload a file in CSV format")

# Upload CSV file
data_file = st.file_uploader("Upload a CSV file", type=['csv'])
if data_file:
    data = pd.read_csv(data_file)
    data['Date'] = parse_dates(data['Date'])
    data.set_index('Date', inplace=True)
    if data.index.freq is None:
        inferred_freq = pd.infer_freq(data.index)
        if inferred_freq:
            data.index = data.index.to_period(inferred_freq).to_timestamp()
        else:
            data.index = data.index.to_period('M').to_timestamp()
    
    # Split data into training and testing sets
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    # Input for forecast horizon
    fh = st.slider('Forecast Horizon', min_value=10, max_value=180, step=10)
    
    # Add a submit button
    if st.button("Submit"):
        with st.spinner('Processing...'):
            best_model, future_forecast, error_metrics = train_model(train_data, fh)
            if future_forecast is not None:
                st.image('forecast_vs_actual.png')
                display_forecast_table(future_forecast)
                
                # Display error metrics
                st.subheader('Model Error Metrics')
                st.dataframe(error_metrics)
            else:
                st.error("No forecast data available.")
