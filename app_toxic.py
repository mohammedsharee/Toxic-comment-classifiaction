# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 18:12:00 2020

@author: Mohammed Shreef
"""

#from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
#import flasgger
import streamlit as st
#from flasgger import Swagger

# app=Flask(__name__)
# Swagger(app)
pickle_in=open('classifier.pkl','rb')
classifier=pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return 'welcome sharif'

#@app.route('/predict',methods=["Get"])



def thermal_comfort_prediction(clo_insulation,metabolic_rate,air_temparature,radiant_temparature,relative_humidity,air_velocity):
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: air_temparature
        in: query
        type: number
        required: true
      - name: radiant_temparature
        in: query
        type: number
        required: true
      - name: relative_humidity
        in: query
        type: number
        required: true
      - name: air_velocity
        in: query
        type: number
        required: true
      - name: clo_insulation
        in: query
        type: number
        required: true
      - name: metabolic_rate
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
   
    prediction=classifier.predict([[clo_insulation,metabolic_rate,air_temparature,radiant_temparature,relative_humidity,air_velocity]])
    print(prediction)
    return prediction


    
def main():
    st.title("Thermal Comfort Prediction")
    html_temp = """
   <body style="background-color:#800080;">
      <div style="background-color:tomato;padding:10px">
      <h2 style="color:white;text-align:center;">Thermal Comfort Prediction ML App </h2>
    </div>
   </body>
    """
    st.markdown(html_temp,unsafe_allow_html=True) 
    
    air_temparature=st.text_input('Air Temparature (°C)')
    radiant_temparature=st.text_input('Radiant Temparature (°C)')
    relative_humidity=st.text_input('Relative Humidity (%)')
    air_velocity=st.text_input('Air Velocity (m/s)')
    clo_insulation=st.text_input('Clo Insulation (Clo)')
    metabolic_rate=st.text_input('Metabolic Rate (Met)')
    result=""
    
    comfortable_html="""
    <div style="background-color:#F4D03F;padding:10px">
    <h2 style="color:white;text-align:center;">Thermally Comfortable [ranges in (-1,0,+1)] </h2>
    </div>
    """
    uncomfortable_html="""
    <div style="background-color:#F00000;padding:10px">
    <h2 style="color:white;text-align:center;">Thermally Uncomfortable [ranges in (-3,-2 and +2,+3)]</h2>
    </div>
    """
    
    
    if st.button("Predict"):
        result=thermal_comfort_prediction(clo_insulation,metabolic_rate,air_temparature,radiant_temparature,relative_humidity,air_velocity)
        st.success('The output is {}'.format(result))       
        if result in range(-1,1,1):
            st.markdown(comfortable_html,unsafe_allow_html=True)
        else:
            st.markdown(uncomfortable_html,unsafe_allow_html=True)
          
    if st.button("About"):
        st.text("This is a thermal comfort prediction app which uses Machine Learning algorithm")
        st.text("Built on Python by using streamlit framework")
        st.text("By Mohammed Shareef")
          

if __name__=='__main__':
    main()