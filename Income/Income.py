# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 17:24:46 2018

@author: bhavesh2429
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd


def normaliseData(x):
  # rescale data to lie between 0 and 1
  scale = x.max(axis=0)
  return (x/scale, scale)

# Load the diabetes dataset
headers = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital-status", "occupation", "relationship", "race",
           "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "class"]

income = pd.read_csv('Income1.csv', header=None, names=headers, na_values="?" )
income.describe()
income.dtypes

#taking objects
obj_df = income.select_dtypes(include=['object']).copy()
obj_df[obj_df.isnull().any(axis=1)]
obj_df["class"].value_counts()
obj_df = obj_df.fillna({"native_country": "United-States"})

obj_df["workclass"].value_counts()
obj_df = obj_df.fillna({"workclass": "Private"})

obj_df["occupation"].value_counts()
obj_df = obj_df.fillna({"occupation": "Prof-specialty"})

obj_df.to_csv('Cleaned_Income1.csv')
#making categories
obj_df["workclass"] = obj_df["workclass"].astype('category')
obj_df.dtypes
obj_df["workclass"] = obj_df["workclass"].cat.codes

obj_df["education"] = obj_df["education"].astype('category')
obj_df.dtypes
obj_df["education"] = obj_df["education"].cat.codes

obj_df["marital-status"] = obj_df["marital-status"].astype('category')
obj_df.dtypes
obj_df["marital-status"] = obj_df["marital-status"].cat.codes

obj_df["occupation"] = obj_df["occupation"].astype('category')
obj_df.dtypes
obj_df["occupation"] = obj_df["occupation"].cat.codes

obj_df["relationship"] = obj_df["relationship"].astype('category')
obj_df.dtypes
obj_df["relationship"] = obj_df["relationship"].cat.codes

obj_df["race"] = obj_df["race"].astype('category')
obj_df.dtypes
obj_df["race"] = obj_df["race"].cat.codes

obj_df["sex"] = obj_df["sex"].astype('category')
obj_df.dtypes
obj_df["sex"] = obj_df["sex"].cat.codes

obj_df["native_country"] = obj_df["native_country"].astype('category')
obj_df.dtypes
obj_df["native_country"] = obj_df["native_country"].cat.codes

obj_df["class"] = obj_df["class"].astype('category')
obj_df.dtypes
obj_df["class"] = obj_df["class"].cat.codes

#Taking integers column
obj_dfint = income.select_dtypes(include=['int64']).copy()
obj_dfint[obj_dfint.isnull().any(axis=1)]
(Xt,Xscale) = normaliseData(obj_dfint)


frames = [Xt, obj_df]
Data1 = pd.concat(frames, axis = 1)



data1 = Data1.iloc[:,:14]
data1
#processing of fnlwgt
data2 = Data1.drop(["native_country", "race"], axis = 1)
data2 = Data1.drop("race", axis = 1)
#############################################


dataip = pd.DataFrame(data1)

dataop = Data1.iloc[:,14:]
from sklearn import preprocessing

data2 = data2.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pandas.DataFrame(x_scaled)

X_train, X_test, y_train, y_test = train_test_split(dataip, dataop, test_size=0.2)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train, test_size=0.6)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn import neighbors, datasets
import matplotlib.pyplot as plt

plt.matshow(Data1.corr())
def correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Abalone Feature Correlation')
    labels=['Sex','Length','Diam','Height','Whole','Shucked','Viscera','Shell','Rings',]
    ax1.set_xticklabels(labels,fontsize=6)
    ax1.set_yticklabels(labels,fontsize=6)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
    plt.show()

correlation_matrix(Data1)

model = LogisticRegression()
model.fit(X_train1, y_train1)
print(model)
# make predictions
expected = y_test
predicted = model.predict(X_test)
accuracy_score(expected, predicted)

# KNN
knn=neighbors.KNeighborsClassifier()
knn.fit(X_train, y_train)
# make predictions
expected = y_test
predicted = knn.predict(X_test)
accuracy_score(expected, predicted)

# decision tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_train, y_train)
print(model)
expected = y_test
predicted = model.predict(X_test)
print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected,predicted))
accuracy_score(expected, predicted)

# random forest
from sklearn.ensemble import RandomForestClassifier
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_train, y_train)
print(model)
expected = y_test
predicted = model.predict(X_test)
print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected,predicted))
accuracy_score(expected, predicted)

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
print(model)
# make predictions
expected = y_test
predicted = model.predict(X_test)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
accuracy_score(expected, predicted)
