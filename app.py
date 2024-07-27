


import pandas as pd
import streamlit as st
from pycaret.time_series import *
from pycaret.datasets import get_data
import codecs
import streamlit.components.v1 as components
from streamlit_lottie import st_lottie
import json
import requests
import numpy as np
from  PIL import Image


@st.experimental_singleton
def load_lottieurl(url:str):
    r = requests.get(url)
    if r.status_code!=200:
        return None
    return r.json()



if "button_clicked" not in st.session_state:
    st.session_state.button_clicked=False

@st.experimental_singleton    
def callback():
    #button was clicked
    st.session_state.button_clicked=True
    

@st.experimental_singleton
def model():
    s=setup(data,fold=3,fh=len(test))  ## Pycaret function
    best=compare_models()
    final_best=finalize_model(best)
    return final_best

    #df=model()

data=get_data('pycaret_downloads')

data['Date']=pd.to_datetime(data['Date'])
data.set_index('Date',inplace=True)
train=int(len(data)*0.8)   ## Defining Training Data for example dataset
train2=data.iloc[:train]  ## Passing Training Data into a variable , First 80% values as training data
test=data.iloc[train:]    ## Passing Tessting Data into a variable , last 20% values as testing data

st.title("Web based Demand Forecasting Application")
st.caption("Use this web based application to forecast your demands - upload a file in CSV format")
lottie_hello=load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_49rdyysj.json")
st_lottie(lottie_hello,speed=1,reverse=True,loop=True,quality="high",height=250,width=1200,key=None)
st.sidebar.title(body="Demand Forecasting Application")
logo = Image.open(r'image3.jpg')
st.sidebar.image(logo,  width=300)

st.sidebar.caption("Ideal for : Demand Prediction on TimeSeries")
st.sidebar.caption("Library Used - PyCaret")
st.sidebar.caption("Designed by : WoodExcavators")

fh=st.slider('How  many days of data you want to predict?',value=10,min_value=10,max_value=180,step=10)

if st.button("Give a try with our example dataset"):
    predict_model(model(),fh=fh)
    plot_model(model(),plot='forecast',data_kwargs={'fh':fh})

else:    
    data_file=st.file_uploader("Want to try this out with your own data ? - upload a csv file here",type=['csv']) 
    st.caption("CSV file should have first column named as Date. Target figures should be in next consecutive column")
    st.caption("Data format for Date Column can be either mm/dd/yyyy or dd/mm/yyy")
    if data_file is not None:
        data_file=pd.read_csv(data_file)
        data_file['Date']=pd.to_datetime(data_file['Date'])
        data_file.set_index('Date',inplace=True)
        train_data=int(len(data_file)*0.8)
        train_data2=data_file.iloc[:train_data]   ## First 80% values as training data
        test_data2=data_file.iloc[train_data:]    ## Last 20% values as testing data
        predict_model(model(),fh=fh)    ## fh figure in predict function tells how many time horizons you want to see ahead
        plot_model(model(),plot='forecast',data_kwargs={'fh':fh})
