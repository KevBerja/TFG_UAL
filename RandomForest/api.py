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

global clf
global model_columns

@app.route('/predict', methods=['POST'])
def predict():
    if clf:
        try:
            json_ = request.json
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(clf.predict(query))

            # Converting to int from int64
            return jsonify({"prediction": list(map(int, prediction))})

        except Exception as e:

            return jsonify({'error': str(e), 'trace': traceback.format_exc()})
    else:
        return '<h1>Upload csv file and training first<h1>'


@app.route('/train', methods=['GET'])
def train():
    df = pd.read_csv('train.csv')
    include = ['Age', 'Sex', 'Embarked', 'Survived']
    dependent_variable = include[-1]
    df_ = df[include]

    categoricals = []  # going to one-hot encode categorical variables

    for col, col_type in df_.dtypes.items():        
        if col_type == 'O':
            categoricals.append(col)
        else:
            df_[col].fillna(0, inplace=True)  # fill NA's with 0 for ints/floats

    # get_dummies effectively creates one-hot encoded variables
    df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)

    x = df_ohe[df_ohe.columns.difference([dependent_variable])]
    y = df_ohe[dependent_variable]

    # capture a list of columns that will be used for prediction
    model_columns = list(x.columns)

    jb.dump(model_columns, 'model_columns.pkl')

    clf = RandomForestClassifier()
    start = time.time()
    clf.fit(x, y)


    jb.dump(clf, 'model.pkl')

    message1 = 'Trained in %.5f seconds' % (time.time() - start)
    message2 = 'Model training score: %s' % clf.score(x, y)
    return_message = 'Success. \n{0}. \n{1}.'.format(message1, message2)

    return "OK"

@app.route('/wipe', methods=['GET'])
def wipe():
    try:
        os.remove('model.pkl')
        os.remove('model_columns.pkl')
        os.removed('train.csv')
        return 'Model removed'

    except Exception as e:
        print(str(e))
        return 'Could not remove the model'

@app.route('/loadCSV', methods=['GET'])
def upload_file():
    return render_template('formulario.html')

@app.route('/uploadCSV', methods=['POST'])
def uploader():
    if request.method == 'POST':
        # obtenemos el archivo del input "archivo"
        file_form = request.files['archivo']
        file = secure_filename(file_form.filename)

        filename = file.split('.')
        f_name = filename[0]
        f_extension = filename[1]

        if (f_name != "train" and f_extension != "csv"):
            return "<h1>Archivo no válido. Subir fichero CSV con nombre -train.csv-<h1>"

        file_form.save(os.path.join('', file))
        # Retornamos una respuesta satisfactoria
        return "<h1>Archivo subido exitosamente</h1>"


if not os.path.exists('model.pkl') and not os.path.exists('model_columns.pkl') and os.path.exists('train.csv'): 
    train()

if os.path.exists('model.pkl') and os.path.exists('model_columns.pkl'):
    # Loading model
    clf = jb.load('model.pkl')
    print ('Model loaded!')
    model_columns = jb.load('model_columns.pkl')
    print ('Model columns loaded!')

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000)