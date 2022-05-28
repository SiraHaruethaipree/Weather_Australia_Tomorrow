import joblib
import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import streamlit as st


random_forest = joblib.load('randomforest.pkl')
sc = joblib.load('scaler.pkl')

st.title("Weather Australia Tomorrow")



st.sidebar.header("Filter data 15 feature")
Humidity3pm = st.sidebar.slider('Humidity3pm:', 0, 10)
Temp3pm = st.sidebar.slider('Temp3pm:', 0, 50)
MaxTemp = st.sidebar.slider('MaxTemp:', 0, 50)
MinTemp = st.sidebar.slider('MinTemp:', -10, 40)
Pressure3pm = st.sidebar.slider('Pressure3pm:', 990, 1050)
Humidity9am = st.sidebar.slider('Humidity9am:', 10, 100)
Temp9am = st.sidebar.slider('Temp9am:', -10, 50)
Pressure9am = st.sidebar.slider('Pressure9am:', 990, 1050)
WindGustSpeed = st.sidebar.slider('WindGustSpeed:', 0, 70)
WindSpeed3pm = st.sidebar.slider('WindSpeed3pm:', 0, 70)
WindSpeed9am = st.sidebar.slider('WindSpeed9am:', 0, 70)
Cloud3pm = st.sidebar.slider('Cloud3pm:', 0, 8)
Rainfall = st.sidebar.slider('Rainfall:', 0, 2)
Cloud9am = st.sidebar.slider('Cloud9am:', 0, 8)
Evaporation = st.sidebar.slider('Evaporation:', 1, 10)

result = random_forest.predict(sc.transform([[Humidity3pm,
 Temp3pm,
 MaxTemp,
 MinTemp,
 Pressure3pm,
 Humidity9am,
 Temp9am,
 Pressure9am,
 WindGustSpeed,
 WindSpeed3pm,
 WindSpeed9am,
 Cloud3pm,
 Rainfall,
 Cloud9am,
 Evaporation]]))

result_prob = random_forest.predict_proba(sc.transform([[Humidity3pm,
 Temp3pm,
 MaxTemp,
 MinTemp,
 Pressure3pm,
 Humidity9am,
 Temp9am,
 Pressure9am,
 WindGustSpeed,
 WindSpeed3pm,
 WindSpeed9am,
 Cloud3pm,
 Rainfall,
 Cloud9am,
 Evaporation]]))

result1 = result[0]


st.header("Result :")
if result1 == 1 :
 original_title = '<p style="font-family: Georgia, serif; color:Blue; font-size: 30px;">Tomorrow is rain</p>'
 st.markdown(original_title, unsafe_allow_html=True)
#st.header("Tomorrow is rain")
else :
 original_title = '<p style="font-family: Georgia, serif; color:Red; font-size: 30px;">Tomorrow is not raining</p>'
 st.markdown(original_title, unsafe_allow_html=True)
 #st.header("Tomorrow is not raining")

st.subheader("Percent Probability")
col1, col2 = st.columns(2)
with col1:
 st.write("Percent not Raining")
 st.write(round((result_prob[0][0])*100,2),"%")
with col2:
 st.write("Percent Raining")
 st.write(round((result_prob[0][1])*100, 2),"%")