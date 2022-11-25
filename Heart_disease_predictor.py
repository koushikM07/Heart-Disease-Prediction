# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 23:33:37 2022

@author: koush
"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('C:/Users/koush/OneDrive/Desktop/Heart Disease/trained_model.sav', 'rb'))

def heart_pred(input_data):
    
    
    
    input_data_as_numpy_array= np.asarray(input_data,dtype=float)

    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    #print(prediciton)
    if (prediction[0] == 0):
      return "The Person does not have any heart disease"
    else:
      return "The person is might in risk"


def main():
    st.title("Heart Disease Predictor")
    
       
    age = st.text_input("Age of Patient")
    sex = st.text_input("Gender of Patient")
    cp = st.text_input("cp")
    trestbps = st.text_input("trestbps")
    chol = st.text_input("chol")
    fbs = st.text_input("fbs")
    restecg = st.text_input("restecg")
    thalach = st.text_input("thalach")
    exang = st.text_input("exang")
    oldpeak = st.text_input("oldpeak")
    slope = st.text_input("slope")
    ca = st.text_input("ca")
    thal = st.text_input("thal")
    
    
    diagnosis = ''
    
    if st.button('Heart_Disease_test_result'):
        diagnosis = heart_pred([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])
    
    st.success(diagnosis)
    
    
if __name__=='__main__':
    main()
    
    
    
    