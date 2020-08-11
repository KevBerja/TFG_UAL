import sys
import os
import shutil
import joblib as jb
import traceback as tr
import pandas as pd
import numpy as np
import mysql.connector

from flask import Flask, request, jsonify, render_template, redirect, session, flash, send_file
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

        
        os.makedirs('./static/model_temp', exist_ok=True)
        file_form.save(os.path.join('', 'static/model_temp/' + file))

        flash("Archivo de datos subido con éxito")

        return redirect('/loadInitCSV')


@app.route('/loadModel', methods=['GET'])
def loadModel():
    try:
        mydb = mysql.connector.connect(
            host="db",
            user="root",
            password="admin",
            database="mlaas")

        mycursor = mydb.cursor()

        mycursor.execute("SELECT nombre, tipo, url FROM modelos WHERE tipo = 'Logistic Regression'")

        myresult = mycursor.fetchall()

        return render_template('cargar_modelo.html', rows=myresult)
    
    except Exception as e:
        flash("ERROR - No se han podido obtener la lista de modelos de la base de datos")

        return render_template('cargar_modelo.html')


@app.route('/setModel', methods=['POST'])
def setModel():
    if request.method == 'POST':
        json_values = list(request.form.values())
        model_name = json_values[0]
        
        try:
            shutil.copy('./static/models/' + model_name + '/train.csv', './static/model_temp/train.csv')
            shutil.copy('./static/models/' + model_name + '/model.pkl', './static/model_temp/model.pkl')
            shutil.copy('./static/models/' + model_name + '/model_columns.pkl', './static/model_temp/model_columns.pkl') 

            flash("Modelo " + model_name + " descargado correctamente")

            return redirect('/loadModel')

        except Exception as e:
            folder_path = './static/model_temp' 
            
            for file_object in os.listdir(folder_path): 
                file_object_path = os.path.join(folder_path, file_object) 
                        
                if os.path.isfile(file_object_path): 
                    os.unlink(file_object_path) 
                else: 
                    shutil.rmtree(file_object_path)

            flash("ERROR - No se ha podido descargar el modelo: " + model_name)

            return redirect('/loadModel')



@app.route('/deleteModel', methods=['GET', 'DELETE'])
def wipe():
    try:
        folder_path = './static/model_temp' 
        
        for file_object in os.listdir(folder_path): 
            file_object_path = os.path.join(folder_path, file_object) 
            
            if os.path.isfile(file_object_path): 
                os.unlink(file_object_path) 
            else: 
                shutil.rmtree(file_object_path)
        
        flash("Modelo eliminado correctamente")
        
        return redirect('/')

    except Exception as e:
        flash("ERROR - No se ha podido eliminar el modelo. Por favor, verifica si el modelo que desea eliminar existe")
        
        return redirect('/')


@app.route('/formTrain', methods=['GET'])
def formTrain():
    return render_template('train.html')

@app.route('/train', methods=['POST'])
def train():
    if request.method == 'POST':
        model = list(request.form.values())
        model_name = model[0]

        if os.path.exists('./static/model_temp/train.csv'):
            if os.path.exists('./static/models/' + model_name):
                flash("Existe un modelo con el nombre '" + model_name + "'. Por favor, utilice otro nombre")
                  
                return redirect('/formTrain')
                
            else:   
                os.makedirs('./static/models/' + model_name, exist_ok=True)

            df = pd.read_csv('./static/model_temp/train.csv', encoding='latin-1')
            include = [str(x) for x in df.columns]  
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

            model_columns = list(x.columns)
            clf = LogisticRegression()
                
            # Test Data and Training Data
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
            clf.fit(x_train, y_train)

            # Accuracy model score
            y_pred = clf.predict(x_test)
            score = accuracy_score(y_pred, y_test)

            try:
                jb.dump(clf, './static/model_temp/model.pkl')
                jb.dump(model_columns, './static/model_temp/model_columns.pkl')

                flash("Modelo entrenado correctamente")
                
            except Exception as e:
                folder_path = './static/model_temp' 
        
                for file_object in os.listdir(folder_path): 
                    file_object_path = os.path.join(folder_path, file_object) 
                    
                    if os.path.isfile(file_object_path): 
                        os.unlink(file_object_path) 
                    else: 
                        shutil.rmtree(file_object_path)

                flash("ERROR - Ha ocurrido un problema durante el entrenamiento")
                        
                return redirect('/formTrain')

            try:
                shutil.copy('./static/model_temp/train.csv', './static/models/' + model_name + '/train.csv')
                shutil.copy('./static/model_temp/model.pkl', './static/models/' + model_name + '/model.pkl')
                shutil.copy('./static/model_temp/model_columns.pkl', './static/models/' + model_name + '/model_columns.pkl')

                try:
                    mydb = mysql.connector.connect(
                        host="db",
                        user="root",
                        password="admin",
                        database="mlaas")

                    mycursor = mydb.cursor()

                    sql = "INSERT INTO modelos (tipo, nombre, url) VALUES (%s, %s, %s)"
                    val = ("Logistic Regression", model_name, './static/models/' + model_name)
                    
                    mycursor.execute(sql, val)

                    mydb.commit()

                    flash("Modelo '" + model_name + "' guardado en la base de datos correctamente")
                                
                    return redirect('/formTrain')
                        
                except Exception as ex:
                    shutil.rmtree('./static/models/' + model_name)

                    flash("ERROR - No se ha podido guardar el modelo: " + model_name +". Por favor, vuelva de nuevo más tarde")

                    return redirect('/formTrain')

                
            except Exception as exc:
                shutil.rmtree('./static/models/' + model_name)

                flash("ERROR - No se ha podido guardar el modelo: " + model_name +". Por favor, vuelva de nuevo más tarde")
                        
                return redirect('/formTrain')

        else:
            flash("Es necesario subir un modelo de datos")

            return redirect('/formTrain')



@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        clf = jb.load('./static/model_temp/model.pkl')
        model_columns = jb.load('./static/model_temp/model_columns.pkl')
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
    try:
        df = pd.read_csv('./static/model_temp/train.csv', encoding='latin-1')
        columns = [str(x) for x in df.columns]
        columns.pop()

        return render_template('formulario_predict.html', columns=columns)
    
    except Exception as e:
        flash("Es necesario subir y entrenar un modelo o cargar un modelo existente")

        return redirect('/')
    

@app.route('/predict_form', methods=['POST'])
def predict_form():
    if request.method == 'POST':
        clf = jb.load('./static/model_temp/model.pkl')
        model_columns = jb.load('./static/model_temp/model_columns.pkl')
        
        if clf:
            try:        
                keys = list(request.form)
                values = list(request.form.values())

                json_ = {}
                iter = 0

                for v in values:
                    if v.isdigit():
                        json_[keys[iter]] = int(v)
                        iter = iter + 1
                        continue
                    if v.find(".") != -1:
                        vx = v.split(".")

                        if vx[0].isdigit() == True and vx[1].isdigit() == True:
                            json_[keys[iter]] = float(v)
                            iter = iter + 1
                            continue
                        else:
                            json_[keys[iter]] = v
                            iter = iter + 1
                            continue 
                    else:
                        json_[keys[iter]] = v
                        iter = iter + 1

                query = [json_]
                query = pd.get_dummies(pd.DataFrame(query))
                query = query.reindex(columns=model_columns, fill_value=0)
                prediction = clf.predict(query)

                df = pd.read_csv('./static/model_temp/train.csv', encoding='latin-1')
                columns = [str(x) for x in df.columns]
                columns.pop()
                
                if prediction[0] == 1:  
                    flash("Predicción realizada con éxito")

                    return render_template('formulario_predict.html', prediction=prediction[0], columns=columns)

                if prediction[0] == 0:
                    flash("Predicción realizada con éxito")

                    return render_template('formulario_predict.html', prediction=prediction[0], columns=columns)                

            except Exception as e:
                flash("ERROR - La predicción falló. Por favor, revise los datos introducidos")

                return redirect('/load_predict_form')

        else:
            flash("ERROR - Debe entrenar primero un modelo")

            return redirect('/load_predict_form')



@app.route('/loadCSVToPredict', methods=['GET'])
def uploadMassive():
    try:
        df = pd.read_csv('./static/model_temp/train.csv', encoding='latin-1')
        columns = [str(x) for x in df.columns]
        columns.pop()

        return render_template('prediccion_masiva.html')
    
    except Exception as e:
        flash("Es necesario subir y entrenar un modelo o cargar un modelo existente")

        return redirect('/')


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
        
        df = pd.read_csv(file, encoding='latin-1')
        include = [str(x) for x in df.columns]  
        df_ = df[include]

        categoricals = []

        for col, col_type in df_.dtypes.items():        
            if col_type == 'O':
                categoricals.append(col)
            else:
                df_[col].fillna(0, inplace=True)

        query = pd.get_dummies(df_, columns=categoricals, dummy_na=True)

        clf = jb.load('./static/model_temp/model.pkl')

        os.remove(file)

        if clf:
            try:
                prediction = list(clf.predict(query))
                prediction_str = [str(i) for i in prediction]
                
                query['predict'] = prediction_str
                query.to_csv('predict.csv', header=True, index=False)

                return send_file('predict.csv', as_attachment=True, attachment_filename='predict.csv')
            
            except Exception as e:
                flash("ERROR - Falló la predicción del fichero de datos")

                return redirect('/loadCSVToPredict')
            
        else:
            flash("ERROR - Es necesario entrenar primero el modelo")
            
            return redirect('/loadCSVToPredict')



if __name__ == '__main__':
    app.run(debug=False, port=5001, host="0.0.0.0")