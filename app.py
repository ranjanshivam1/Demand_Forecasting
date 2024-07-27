import streamlit as st
import pandas as pd
import numpy as np
from pycaret.time_series import *
import time
from PIL import Image

# Add the title and side images
st.title("Web Based Demand Forecasting Application")
st.caption("Upload your CSV file for demand forecasting")
st.sidebar.title("Demand Forecasting Application")
logo = Image.open('image3.jpg')
st.sidebar.image(logo, width=300)
st.sidebar.caption("Library Used - PyCaret")
st.sidebar.caption("Designed by: Your Name")

# User input for number of periods
fh = st.number_input("Enter the number of periods to forecast", min_value=1, max_value=365, value=10)

# Upload file
data_file = st.file_uploader("Upload a CSV file", type=['csv'])

# Function to calculate and display metrics
def calculate_metrics(y_true, y_pred, start_time):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true))
    smape = np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    tt = time.time() - start_time  # Example, adjust based on your timing needs
    return [mae, rmse, mape, smape, r2, tt]

# Button for submission
if st.button("Submit"):
    if data_file:
        with st.spinner('Processing Your Request...'):
            data = pd.read_csv(data_file)
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            train_size = int(len(data) * 0.8)
            train_data = data.iloc[:train_size]
            test_data = data.iloc[train_size:]
            
            # Train models
            start_time = time.time()
            s = setup(train_data, fh=fh, fold=3)
            models_to_compare = ['ets', 'arima', 'prophet', 'naive', 'exp_smooth', 'polytrend', 'croston', 'gbr_cds_dt']
            best_model = compare_models(include=models_to_compare)
            final_best = finalize_model(best_model)
            future_forecast = predict_model(final_best, fh=fh)
            
            # Calculate metrics for each model
            metrics = {}
            for model_name in models_to_compare:
                model = create_model(model_name)
                final_model = finalize_model(model)
                y_pred = predict_model(final_model, data=test_data)['Label']
                y_true = test_data.values
                metrics[model_name] = calculate_metrics(y_true, y_pred, start_time)
            
            # Convert metrics to DataFrame
            metrics_df = pd.DataFrame(metrics, index=['MAE', 'RMSE', 'MAPE', 'SMAPE', 'R2', 'TT (Sec)']).T
            
            # Display metrics
            st.write("Model Metrics:")
            st.dataframe(metrics_df)
    else:
        st.write("Please upload a CSV file to proceed.")
