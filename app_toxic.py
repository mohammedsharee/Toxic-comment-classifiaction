# -*- coding: utf-8 -*-
"""
Created on Thu feb 26 18:12:00 2021

@author: Mohammed Shareef
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
# pickle_in=open('classifier.pkl','rb')
# classifier=pickle.load(pickle_in)


# tox_vec=open('toxic_dict.pkl','rb')
# tox = pickle.load(tox_vec)

# severe_toxic_dict=open("severe_toxic_dict.pkl", "rb") 
# sev = pickle.load(severe_toxic_dict)

# obscene_dict= open("obscene_dict.pkl", "rb") 
# obs = pickle.load(obscene_dict)

# insult_dict=open(r"insult_dict.pkl", "rb")
# ins = pickle.load(insult_dict)

# threat_dict=open(r"threat_dict.pkl", "rb")
# thr = pickle.load(threat_dict)

# identity_hate_dict=open("identity_hate_dict.pkl", "rb")
# ide = pickle.load(identity_hate_dict)

# # Load the pickled RDF models
# toxic_model=open("toxic_model.pkl", "rb")
# tox_model = pickle.load(toxic_model)

# severe_toxic_model=open("severe_toxic_model.pkl", "rb")
# sev_model = pickle.load(severe_toxic_model)

# obscene_model=open("obscene_model.pkl", "rb")
# obs_model  = pickle.load(obscene_model)

# insult_model=open("insult_model.pkl", "rb")
# ins_model  = pickle.load(insult_model)

# threat_model=open("threat_model.pkl", "rb")
# thr_model  = pickle.load(threat_model)

# identity_hate_model=open("identity_hate_model.pkl", "rb")
# ide_model  = pickle.load(identity_hate_model)

with open(r"toxic_dict.pkl", "rb") as f:
    tox = pickle.load(f)

with open(r"severe_toxic_dict.pkl", "rb") as f:
    sev = pickle.load(f)

with open(r"obscene_dict.pkl", "rb") as f:
    obs = pickle.load(f)

with open(r"insult_dict.pkl", "rb") as f:
    ins = pickle.load(f)

with open(r"threat_dict.pkl", "rb") as f:
    thr = pickle.load(f)

with open(r"identity_hate_dict.pkl", "rb") as f:
    ide = pickle.load(f)

# Load the pickled RDF models
with open(r"toxic_model.pkl", "rb") as f:
    tox_model = pickle.load(f)

with open(r"severe_toxic_model.pkl", "rb") as f:
    sev_model = pickle.load(f)

with open(r"obscene_model.pkl", "rb") as f:
    obs_model  = pickle.load(f)

with open(r"insult_model.pkl", "rb") as f:
    ins_model  = pickle.load(f)

with open(r"threat_model.pkl", "rb") as f:
    thr_model  = pickle.load(f)

with open(r"identity_hate_model.pkl", "rb") as f:
    ide_model  = pickle.load(f)
 
   


 
   

#@app.route('/')
def welcome():
    return 'welcome sharif'

#@app.route('/predict',methods=["Get"])



def toxic_classifier(data):
    
    # Take a string input from user
    #user_input = request.form['text']
    data = [data]

    vect = tox.transform(data)
    pred_tox = tox_model.predict_proba(vect)[:,1]

    vect = sev.transform(data)
    pred_sev = sev_model.predict_proba(vect)[:,1]

    vect = obs.transform(data)
    pred_obs = obs_model.predict_proba(vect)[:,1]

    vect = thr.transform(data)
    pred_thr = thr_model.predict_proba(vect)[:,1]

    vect = ins.transform(data)
    pred_ins = ins_model.predict_proba(vect)[:,1]

    vect = ide.transform(data)
    pred_ide = ide_model.predict_proba(vect)[:,1]

    out_tox = 'Probability of Toxic in %: {}'.format(round(pred_tox[0], 2)*100)
    out_sev = 'Probability of Severe Toxic in %: {}'.format(round(pred_sev[0], 2)*100)
    out_obs = 'Probability of Obscene in %: {}'.format(round(pred_obs[0], 2)*100)
    out_ins = 'Probability of Insult in %: {}'.format(round(pred_ins[0], 2)*100)
    out_thr = 'Probability of Threat in %:  {}'.format(round(pred_thr[0], 2)*100)
    out_ide = 'Probability of Identity Hate in %: {}'.format(round(pred_ide[0], 2)*100)

    print(out_tox)

    
    return out_tox, out_sev, out_obs, out_ins, out_thr, out_ide


    
def main():
    st.title("Toxic Comment Classifier")
    html_temp = """
   <body style="background-color:#800080;">
      <div style="background-color:tomato;padding:10px">
      <h2 style="color:white;text-align:center;">Toxic Comment Classifier ML App </h2>
    </div>
   </body>
    """
    st.markdown(html_temp,unsafe_allow_html=True) 
    
    text_data=st.text_input('Please enter the text data (Note: This model takes some time to predict the output)')
    
    result=""

    if st.button("Predict"):
        result=toxic_classifier(text_data)
        st.success('The output is {}'.format(result))    
        
          
    if st.button("About"): 
        st.text("This is a toxic comment prediction app which uses Machine Learning algorithm")
        st.text("Built on Python by using streamlit framework")
        st.text("By Mohammed Shareef")
          

if __name__=='__main__':
    main()
