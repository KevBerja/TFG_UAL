# Importing dependencies
import pandas as pd 
import numpy as np
import re
import string
import joblib as jb

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# # Preproccesing text methods
def remove_pattern(input_txt,pattern):
    r = re.findall(pattern,input_txt)
    for i in r:
        input_txt = re.sub(i,'',input_txt)

    return input_txt

def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    
    return round(count/(len(text) - text.count(" ")),3)*100

# Reading CSV data and loading the Dataset in a Dataframe Object
data = pd.read_csv("train.csv", sep = "\t")

# Data Preprocesing
data.columns = ["label","body_text"]
data['label'] = data['label'].map({'pos': 0, 'neg': 1})
data['tidy_text'] = np.vectorize(remove_pattern)(data['body_text'],"@[\w]*")
tokenized_text = data['tidy_text'].apply(lambda x: x.split())
stemmer = PorterStemmer()
tokenized_text = tokenized_text.apply(lambda x: [stemmer.stem(i) for i in x]) 

for i in range(len(tokenized_text)):
    tokenized_text[i] = ' '.join(tokenized_text[i])

data['tidy_text'] = tokenized_text
data['body_len'] = data['body_text'].apply(lambda x:len(x) - x.count(" "))
data['punct%'] = data['body_text'].apply(lambda x:count_punct(x))
X = data['tidy_text']
y = data['label']

# Extracting feature with CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X) # Fitting the Data
X = pd.concat([data['body_len'],data['punct%'],pd.DataFrame(X.toarray())],axis = 1)

# Test Data and Training Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Using classifier Naive Bayes
clf = MultinomialNB()

# Fitting the model 
clf.fit(X_train,y_train)

# Accuracy model score
y_pred = clf.predict(X_test)
score = accuracy_score(y_pred,y_test)
print(score)

# Saving the trained model
jb.dump(clf, 'model.pkl')
print("Model saved!")
jb.dump(cv, 'cv_model.pkl')
print("Count Vectorizer Model saved!")