# Import dependencies
import pandas as pd 
import numpy as np
import re
import string
import joblib as jb
import traceback as tr

from flask import Flask,render_template,url_for,request, jsonify
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Preproccesing text methods
def remove_pattern(input_txt,pattern):
    r = re.findall(pattern,input_txt)

    for i in r:
        input_txt = re.sub(i,'',input_txt)
    
    return input_txt

def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])

    return round(count/(len(text) - text.count(" ")),3)*100

# Loading model
clf = jb.load('model.pkl')
print("Model loaded!")
cv = jb.load('cv_model.pkl');
print("Count Vectorizer model loaded!")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if clf:
        try:
            message = request.form['message']
            data = [message]
            vect = pd.DataFrame(cv.transform(data).toarray())
            body_len = pd.DataFrame([len(data) - data.count(" ")])
            punct = pd.DataFrame([count_punct(data)])
            total_data = pd.concat([body_len,punct,vect],axis = 1)
            my_prediction = clf.predict(total_data)
            
            return render_template('result.html',prediction = my_prediction)
        
        except:

            return jsonify({'trace': tr.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000)