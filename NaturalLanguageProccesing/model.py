# Importing dependencies
import pandas as pd
import numpy as np
import joblib as jb

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Reading CSV data and loading the Dataset in a Dataframe Object
df = pd.read_csv('train.csv', encoding="latin-1")

# Data Preprocesing
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df['label'] = df['class'].map({'ham': 0, 'spam': 1})
X = df['message']
y = df['label']

# CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)

# Test Data and Training Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Naive Bayes Classifier
clf = MultinomialNB()

# Fitting the model 
clf.fit(X_train,y_train)

# Accuracy model score
y_pred = clf.predict(X_test)
score = accuracy_score(y_pred, y_test)
print(score)

# Saving the trained model
jb.dump(clf, 'model.pkl')
print('Model saved!')
jb.dump(cv, 'cv_model.pkl')
print('Count Vectorizer Model saved!')