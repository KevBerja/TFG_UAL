# Importind dependencies
import joblib as jb
import pandas as pd 
import traceback as tr

from flask import Flask,render_template,url_for,request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Loadind model
clf = jb.load('model.pkl')
print('Model loaded!')
cv = jb.load('cv_model.pkl')
print('Count Vectorizer model loaded!')

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
	if clf:
		try:
			message = request.form['message']
			data = [message]
			vect = cv.transform(data).toarray()
			my_prediction = clf.predict(vect)
			
			return render_template('result.html', prediction = my_prediction)

		except:
			return jsonify({'trace': tr.format_exc()})
    
	else:
	    print('Train the model first')
	    return('No model here to use')

if __name__ == '__main__':
	app.run(host = '0.0.0.0', port = 5000)