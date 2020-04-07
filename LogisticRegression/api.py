# Importing 
import sys
import os
import joblib as jb
import traceback as tr
import pandas as pd
import numpy as np

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)


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
            # Reading CSV data and loading the Dataset in a Dataframe Object
            df = pd.read_csv('train.csv', encoding='latin-1')
            include = ['Age', 'Sex', 'Embarked', 'Survived']
            dependent_variable = include[-1]
            df_ = df[include]

            # Data Preprocessing - Categoricals are the columns that are column types with values which select a type 
            categoricals = []
            for col, col_type in df_.dtypes.iteritems():
                if col_type == 'O':
                    categoricals.append(col)
                else:
                    df_[col].fillna(0, inplace=True)

            df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)

            # Reselecting the training data which were proccesed
            X = df_ohe[df_ohe.columns.difference([dependent_variable])]
            y = df_ohe[dependent_variable]

            # Logistic Regression classifier
            clf = LogisticRegression()

            # Test Data and Training Data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

            # Fitting the model
            clf.fit(X_train, y_train)

            # Accuracy model score
            y_pred = clf.predict(X_test)
            score = accuracy_score(y_pred, y_test)

            # Saving the trained model
            jb.dump(clf, 'model.pkl')

            # Saving the data columns from training
            model_columns = list(X_train.columns)
            jb.dump(model_columns, 'model_columns.pkl')

            return "Model trained correctly!"
        
        else:
            return "You need to upload train.csv file first!"
    
    else:
        if os.path.exists('model.pkl') and os.path.exists('model_columns.pkl'):
            # Loading model
            clf = jb.load('model.pkl')
            model_columns = jb.load('model_columns.pkl')

            return "Model charged correctly!"

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        
        clf = jb.load('model.pkl')
        model_columns = jb.load('model_columns.pkl')

        if clf:
            try:
                json_ = request.json
                query = pd.get_dummies(pd.DataFrame(json_))
                query = query.reindex(columns=model_columns, fill_value=0)
                prediction = list(clf.predict(query))
                prediction_str = [str(i) for i in prediction] 
                return jsonify({'prediction': prediction_str}) 

            except Exception as e:

                return jsonify({'trace': tr.format_exc()})
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
        
        df = pd.read_csv(file)
        include = ['Age', 'Sex', 'Embarked']
        df_ = df[include]

        categoricals = []

        for col, col_type in df_.dtypes.items():        
            if col_type == 'O':
                categoricals.append(col)
            else:
                df_[col].fillna(0, inplace=True)

        query = pd.get_dummies(df_, columns=categoricals, dummy_na=True)
        
        clf = jb.load('model.pkl')
        model_columns = jb.load('model_columns.pkl')

        if clf:
            try:
                prediction = list(clf.predict(query))
                prediction_str = [str(i) for i in prediction] 
                return jsonify({'prediction': prediction_str}) 

            except Exception as e:

                return jsonify({'trace': tr.format_exc()})
        else:

            return "Upload csv file and then train the model!"

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 4000)