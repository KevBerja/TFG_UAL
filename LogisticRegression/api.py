# Importing dependencies
from flask import Flask, request, jsonify
import joblib as jb
import traceback as tr
import pandas as pd
import numpy as np

app = Flask(__name__)

# Loading model
clf = jb.load('model.pkl')
print ('Model loaded!')
model_columns = jb.load('model_columns.pkl')
print ('Model columns loaded!')

@app.route('/predict', methods=['POST'])
def predict():
    if clf:
        try:
            json_ = request.json
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)
            prediction = list(clf.predict(query))
            prediction_str = [str(i) for i in prediction] 
            return jsonify({'prediction': prediction_str}) 

        except:

            return jsonify({'trace': tr.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000)