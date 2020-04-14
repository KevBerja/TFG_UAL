import sys
import os
import traceback
import numpy as np
import pandas as pd
import joblib as jb

from sklearn.linear_model import LinearRegression
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/loadInitCSV', methods=['GET'])
def upload_file():
    return render_template('uploadCSVForTraining.html')

@app.route('/uploadInitCSV', methods=['POST'])
def uploader():
    if request.method == 'POST':
        file_form = request.files['file_request']
        file = secure_filename(file_form.filename)

        filename = file.split('.')
        f_name = filename[0]
        f_extension = filename[-1]

        if (f_name != "train" and f_extension != "csv"):
            return "File not valid. Upload CSV file with the name --train.csv--!"

        file_form.save(os.path.join('', file))

        return "Upload file correctly!"


@app.route('/deleteModel', methods=['POST'])
def wipe():
    if request.method == 'POST':
        try:
            os.remove('model.pkl')
            os.remove('model_columns.pkl')
            os.removed('train.csv')
            return "Model removed correctly!"

        except Exception as e:
            return "Could not remove the model!"


@app.route('/train', methods=['GET'])
def train():    
    if not os.path.exists('model.pkl') and not os.path.exists('model_columns.pkl'):
        if os.path.exists('train.csv'):    
            dataset = pd.read_csv('train.csv')
            dataset['experience'].fillna(0, inplace=True)
            dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

            X = dataset.iloc[:, :3]
            y = dataset.iloc[:, -1]

            model_columns = list(X.columns)
            jb.dump(model_columns, 'model_columns.pkl')

            clf = LinearRegression()
            clf.fit(X, y)

            jb.dump(clf, 'model.pkl')

            return "Model trained correctly!"
        
        else:
            return "You need to upload train.csv file first!"
    
    else:
        if os.path.exists('model.pkl') and os.path.exists('model_columns.pkl'):
            # Loading model
            clf = jb.load('model.pkl')
            model_columns = jb.load('model_columns.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        clf = jb.load('model.pkl')
        model_columns = jb.load('model_columns.pkl')
        if clf:
            try:
                json_ = request.json
                query = pd.DataFrame(json_)
                query = query.reindex(columns=model_columns, fill_value=0)
                prediction = clf.predict(query)
                output = round(prediction[0], 2)

                return jsonify({"prediction": str(output)})
            except Exception as e:

                return jsonify({'error': str(e), 'trace': traceback.format_exc()})
        else:
            return "Upload csv file and then train the model!"


@app.route('/predict_form', methods=['POST'])
def predict_form():
    if request.method == 'POST':
        clf = jb.load('model.pkl')
        model_columns = jb.load('model_columns.pkl')
        if clf:
            try:
                json_ = [float(x) for x in request.form.values()]
                query = [np.array(json_)]
                prediction = clf.predict(query)
                output = round(prediction[0], 2)

                return render_template('index.html', prediction='Employee Salary should be $ {}'.format(str(output)))

            except Exception as e:

                return jsonify({'error': str(e), 'trace': traceback.format_exc()})
        else:
            return "Upload csv file and then train the model!"


@app.route('/loadCSVToPredict', methods=['GET'])
def uploadMassive():
    return render_template('uploadCSVToPredict.html')

@app.route('/predictMassive', methods=['POST'])
def predictMassive():
    if request.method == 'POST':
        file_form = request.files['file_request']
        file = secure_filename(file_form.filename)
        filename = file.split('.')
        f_extension = filename[-1]

        if (f_extension != "csv"):
            return "File not allowed. Upload a CSV file valid!"

        file_form.save(os.path.join('', file))
        
        dataset = pd.read_csv(file)
        dataset['experience'].fillna(0, inplace=True)
        dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

        query = dataset.iloc[:, :3]

        clf = jb.load('model.pkl')
        model_columns = jb.load('model_columns.pkl')

        os.remove(file)

        if clf:
            try:
                prediction = list(clf.predict(query))
                prediction_float = [float(x) for x in prediction]
                prediction_round = [round(x, 2) for x in prediction_float]
                prediction_str = [str(x) for x in prediction_round]

                return jsonify({"prediction": prediction_str})
            
            except Exception as e:

                return jsonify({'error': str(e), 'trace': traceback.format_exc()})
            
        else:
            return "You need to train the model first!"
    
        

if __name__ == "__main__":
    app.run(host = '0.0.0.0', port = 5000)