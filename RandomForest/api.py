import sys
import os
import time
import traceback
import pandas as pd
import joblib as jb

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from sklearn.ensemble import RandomForestClassifier

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
            df = pd.read_csv('train.csv', encoding='latin-1')
            include = ['Age', 'Sex', 'Embarked', 'Survived']
            dependent_variable = include[-1]
            df_ = df[include]

            categoricals = []

            for col, col_type in df_.dtypes.items():        
                if col_type == 'O':
                    categoricals.append(col)
                else:
                    df_[col].fillna(0, inplace=True)

            df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)

            x = df_ohe[df_ohe.columns.difference([dependent_variable])]
            y = df_ohe[dependent_variable]

            # capture a list of columns that will be used for prediction
            model_columns = list(x.columns)

            jb.dump(model_columns, 'model_columns.pkl')

            clf = RandomForestClassifier()
            clf.fit(x, y)

            jb.dump(clf, 'model.pkl')

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

                return jsonify({'error': str(e), 'trace': traceback.format_exc()})
            
        else:
            return "You need to train the model first!"
    
        os.remove(file)

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000)