# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 17:24:46 2018

@author: bhavesh2429
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn import neighbors, datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier




def normaliseData(x):
  # rescale data to lie between 0 and 1
  scale = x.max(axis=0)
  return (x/scale, scale)

def con_category(a):
  obj_df[a] = obj_df[a].astype('category')
  obj_df.dtypes
  obj_df[a] = obj_df[a].cat.codes

# Load the income dataset
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

#filling null values
obj_df = obj_df.fillna({"native_country": "United-States"})

obj_df["workclass"].value_counts()
obj_df = obj_df.fillna({"workclass": "Private"})
obj_df["occupation"].value_counts()
obj_df = obj_df.fillna({"occupation": "Prof-specialty"})

obj_df.to_csv('Cleaned_Income1.csv')
#making categories
con_category("workclass")
con_category("education")
con_category("marital-status")
con_category("occupation")
con_category("relationship")
con_category("race")
con_category("sex")
con_category("native_country")
con_category("class")

#Taking integers column
obj_dfint = income.select_dtypes(include=['int64']).copy()
obj_dfint[obj_dfint.isnull().any(axis=1)]
(Xt,Xscale) = normaliseData(obj_dfint)

#combining category and integer columns
frames = [Xt, obj_df]
Data1 = pd.concat(frames, axis = 1)

#splitting a dataframe to form input 
data1 = Data1.iloc[:,:14]
data1
#removal of education column
data1 = data1.drop(["education"], axis = 1)

#inputs
dataip = pd.DataFrame(data1)
#outputs
dataop = Data1.iloc[:,14:]
#Uncomment this line for data size = 16k
X_train1, X_test1, y_train1, y_test1 = train_test_split(dataip, dataop, train_size=0.8)

#executing various models for different values of input
for j in range(1, 6, 1):
   f1_acc = [] 
   count = []
   for i in range(1, 20, 1):
      a = i / 20.0
      #uncomment this line for 16k rows
      X_train, X_test, y_train, y_test = train_test_split(X_train1, y_train1, train_size=a)
      #X_train, X_test, y_train, y_test = train_test_split(dataip, dataop, train_size=a)

      n=len(X_train)
#logistic regression
      if j == 1:
         save = 'Logistic.png'
         model = LogisticRegression(intercept_scaling=2, max_iter=1000)
         model.fit(X_train, y_train)
      if j == 2:
#KNN
         knn=neighbors.KNeighborsClassifier()
         knn.fit(X_train, y_train)
         save = 'KNN.png'
#Decision Tree
      if j == 3:
         model = DecisionTreeClassifier(criterion='entropy', min_samples_split=8)
         model.fit(X_train, y_train)
         save = 'DT.png'
# print(model)
#random forest
      if j == 4:
         model = DecisionTreeClassifier(criterion='entropy', min_samples_split=8)
         model.fit(X_train, y_train)
         save = 'RF.png'
      if j == 5:
#Naive Bayes
        model = GaussianNB()
        model.fit(X_train, y_train)
        save = 'NB.png'
#print(model)
# make predictions
      expected = y_test1
      predicted = model.predict(X_test1)
#predicted=model.predict(X_test[:,0:-1])
      accuracy1 = f1_score(expected, predicted, average = 'weighted')
      f1_acc.append(accuracy1)
      count.append(n)
#plotting graphs
      fig, ax = plt.subplots(figsize=(12,8))
      ax.scatter(count, f1_acc, c = 'b', marker = 'o', label = 'F1 Accuracy')
      ax.set_title(save)
      ax.legend()
      ax.set_xlabel('Dataset Size')
      ax.set_ylabel('Accuracy')
      fig.savefig(save)
      

  
