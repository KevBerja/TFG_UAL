# Importing 
import sys
import os
import joblib as jb
import traceback as tr
import pandas as pd
import numpy as np

from flask import Flask, request, jsonify, render_template, redirect, session, flash
from werkzeug.utils import secure_filename
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)
app.secret_key = 'kcv239LogisticRegression'


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


@app.route('/deleteModel', methods=['GET'])
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
            # Reading CSV data and loading the Dataset in a Dataframe Object
            df = pd.read_csv('train.csv', encoding='latin-1')
            include = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
            dependent_variable = include[-1]

            # Reselecting the training data which were proccesed
            X = df[df.columns.difference([dependent_variable])]
            y = df[dependent_variable]

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

            flash("Modelo entrenado correctamente")

            return redirect('/')
        
        else:
            flash("ERROR - Se necesita primero subir el fichero de datos para entrenamiento")
            
            return redirect('/')
    
    else:
        if os.path.exists('model.pkl') and os.path.exists('model_columns.pkl'):
            # Loading model
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

                return jsonify({"prediction": str(prediction)})
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
                json_ = list(request.form.values())
                query = [{'Pregnancies': int(json_[0]), 'Glucose': int(json_[1]), 'BloodPressure': int(json_[2]), 'SkinThickness': int(json_[3]), 'Insulin': int(json_[4]) ,'BMI': float(json_[5]), 'DiabetesPedigreeFunction': float(json_[6]), 'Age': int(json_[7])}]
                query = pd.get_dummies(pd.DataFrame(query))
                query = query.reindex(columns=model_columns, fill_value=0)
                prediction = clf.predict(query)
                
                if prediction[0] == 1:  
                    flash("Predicción realizada con éxito")

                    return render_template('formulario_predict.html', prediction='Diabetes: Sí')

                if prediction[0] == 0:
                    flash("Predicción realizada con éxito")

                    return render_template('formulario_predict.html', prediction='Diabetes: No')                

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
        
        df = pd.read_csv(file)

        clf = jb.load('model.pkl')
        model_columns = jb.load('model_columns.pkl')

        query = df.iloc[:, :8]

        os.remove(file)

        if clf:
            try:
                prediction = list(clf.predict(query))
                prediction_str = [str(i) for i in prediction] 
                return jsonify({'prediction': prediction_str}) 

            except Exception as e:

                flash("ERROR - Falló la predicción del fichero de datos")

                return redirect('/loadCSVToPredict')
        else:
                flash("ERROR - Es necesario entrenar primero el modelo")
            
                return redirect('/loadCSVToPredict')

if __name__ == '__main__':
    app.run(debug=False, port=5001, host="0.0.0.0")