from flask import Flask,render_template,url_for,request
import pickle
import numpy as np
import joblib
import pandas as pd
app=Flask(__name__)

model_path = 'pipeline.pkl'


model = joblib.load(
    open(model_path,'rb'))
print("model read")
@app.route('/')

def home():
    return render_template('home.html')

@app.route('/result',methods=['POST'])
def predict():
  
    text_URL = (request.form['text_URL'])
    print(text_URL)
    prediction=model.predict([text_URL])
   
   
    return render_template('result.html',prediction=prediction)
   
if __name__=='__main__':
    app.run(debug=True)
    
                       
    
    














