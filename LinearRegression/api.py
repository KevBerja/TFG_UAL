import sys
import os
import traceback as tr
import numpy as np
import pandas as pd
import joblib as jb

from flask import Flask, request, jsonify, render_template, redirect, session, flash
from werkzeug.utils import secure_filename
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


app = Flask(__name__)
app.secret_key = 'kcv239LinearRegression'


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/loadInitCSV', methods=['GET'])
def upload_file():
    return render_template('subida_fichero.html')

@app.route('/uploadInitCSV', methods=['POST'])
def uploader():
    if request.method == 'POST':
        file_form = request.files['file_request']
        file = secure_filename(file_form.filename)

        filename = file.split('.')
        f_name = filename[0]
        f_extension = filename[-1]

        if (f_name != "train" and f_extension != "csv"):

            flash("ERROR - Archivo no válido. El fichero de datos de entrenamiento debe estar en formato .csv con el nombre --train.csv--")
            
            return redirect('/loadInitCSV')

        file_form.save(os.path.join('', file))

        flash("Archivo de datos subido con éxito")

        return redirect('/loadInitCSV')


@app.route('/deleteModel', methods=['GET', 'DELETE'])
def wipe():
    try:
        os.remove('model.pkl')
        os.remove('model_columns.pkl')
        
        flash("Modelo eliminado correctamente")
        
        return redirect('/')

    except Exception as e:
        flash("ERROR - No se ha podido eliminar el modelo. Por favor, verifica si el modelo que desea eliminar existe")
        
        return redirect('/')


@app.route('/train', methods=['GET'])
def train():    
    if not os.path.exists('model.pkl') and not os.path.exists('model_columns.pkl'):
        if os.path.exists('train.csv'):    
            dataset = pd.read_csv('train.csv')
            dataset['experience'].fillna(0, inplace=True)
            dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

            x = dataset.iloc[:, :3]
            print(x)
            y = dataset.iloc[:, -1]
            print(y)

            model_columns = list(x.columns)
            jb.dump(model_columns, 'model_columns.pkl')

            clf = LinearRegression()
            clf.fit(x, y)

            jb.dump(clf, 'model.pkl')

            flash("Modelo entrenado correctamente")

            return redirect('/')

        
        else:
            flash("ERROR - Se necesita primero subir el fichero de datos para entrenamiento")
            
            return redirect('/')
    
    else:
        clf = jb.load('model.pkl')
        model_columns = jb.load('model_columns.pkl')
            
        flash("Modelo entrenado correctamente")
            
        return redirect('/')


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

                return jsonify({'error': str(e), 'trace': tr.format_exc()})
        else:
            return "ERROR - Se necesita primero subir el fichero de datos para entrenamiento y entrenar después al modelo"


@app.route('/load_predict_form', methods=['GET'])
def load_form():
    return render_template('formulario_predict.html')

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

                flash("Predicción realizada con éxito")

                return render_template('formulario_predict.html', prediction="El total estimado de salario para el empleado es de {} euros".format(str(output)))

            except Exception as e:

                flash("ERROR - La predicción falló. Por favor, revise los datos introducidos")

                return redirect('/load_predict_form')
        else:
            
            flash("ERROR - Debe entrenar primero un modelo")

            return redirect('/load_predict_form')


@app.route('/loadCSVToPredict', methods=['GET'])
def uploadMassive():
    return render_template('prediccion_masiva.html')

@app.route('/predictMassive', methods=['POST'])
def predictMassive():
    if request.method == 'POST':
        file_form = request.files['file_request']
        file = secure_filename(file_form.filename)
        filename = file.split('.')
        f_extension = filename[-1]

        if (f_extension != "csv"):
            flash("ERROR - Archivo no válido. Suba un fichero en formato .csv correcto")
            
            return redirect('/loadCSVToPredict')

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
                flash("ERROR - Falló la predicción del fichero de datos")

                return redirect('/loadCSVToPredict')
            
        else:
            flash("ERROR - Es necesario entrenar primero el modelo")
            
            return redirect('/loadCSVToPredict')
    
        

if __name__ == "__main__":
    app.run(debug=False, port=5000, host="0.0.0.0")