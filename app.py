# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:16:36 2021

@author: Guttula Tejovani
"""

import numpy as np
from flask import Flask,request,render_template
from joblib import load
import joblib
from keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer
import keras
import pickle
import tensorflow as tf

tf.keras.backend.clear_session() 

model=tf.keras.models.load_model("amazon.h5")

app= Flask(__name__) 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    d=request.form['Sentence']
    print(d) 
    with open(r'CountVectorizer','rb') as file:
        cv=pickle.load(file)
        d=cv.transform([d])
    result=model.predict(d)
    print(result)
    prediction=result>0.5
    if prediction[0]==False:
        output="Positive Review"
    elif prediction[0]==True:
        output="Negative Review"
    return render_template('index.html',prediction_text=str(output)) 

if __name__=="__main__":
    app.run(debug=True) 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    