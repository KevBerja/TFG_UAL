import numpy as np
import pandas as pd
import joblib as jb

from sklearn.linear_model import LinearRegression
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


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

model = jb.load('model.pkl')
model_columns = jb.load('model_columns.pkl')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    json_ = request.json
    query = pd.DataFrame(json_)
    query = query.reindex(columns=model_columns, fill_value=0)
    prediction = model.predict(query)
    output = round(prediction[0], 2)

    return jsonify({"prediction": str(output)})


@app.route('/predict_form',methods=['POST'])
def predict_form():
    json_ = [float(x) for x in request.form.values()]
    query = [np.array(json_)]
    prediction = model.predict(query)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction='Employee Salary should be $ {}'.format(str(output)))

if __name__ == "__main__":
    app.run(host = '0.0.0.0', port = 5000)