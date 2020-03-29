# Importing dependencies
import pandas as pd
import numpy as np
import joblib as jb

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Reading CSV data and loading the Dataset in a Dataframe Object
df = pd.read_csv('train.csv', encoding="latin-1")
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

# Dependent variable
# dependent_variable = 'Survived'

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
print(score)

# Saving the trained model
jb.dump(clf, 'model.pkl')
print("Model saved!")

# Saving the data columns from training
model_columns = list(X_train.columns)
jb.dump(model_columns, 'model_columns.pkl')
print("Models columns saved!")