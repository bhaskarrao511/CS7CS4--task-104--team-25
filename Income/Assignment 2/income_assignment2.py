# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 17:24:46 2018

@author: bhavesh2429
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC  


# rescale data to lie between 0 and 1
def normaliseData(x):
  scale = x.max(axis=0)
  return (x/scale, scale)

# create a category
def con_category(a):
  obj_df[a] = obj_df[a].astype('category')
  obj_df.dtypes
  obj_df[a] = obj_df[a].cat.codes

# Load the income dataset
headers = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital-status", "occupation", "relationship", "race",
           "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "class"]

income = pd.read_csv('Income.csv', header=None, names=headers, na_values="?" )
income.describe()
income.dtypes

# dropping null value rows
income1 = income.dropna(axis=0, subset=['workclass'])
#taking objects
obj_df = income1.select_dtypes(include=['object']).copy()
obj_df[obj_df.isnull().any(axis=1)]
obj_df["class"].value_counts()

#filling null values with a new category
obj_df = obj_df.fillna({"native_country": "Unkonown"})

obj_df["occupation"].value_counts()
obj_df = obj_df.fillna({"occupation": "New_Profession"})


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
obj_dfint = income1.select_dtypes(include=['int64']).copy()
obj_dfint[obj_dfint.isnull().any(axis=1)]
test1 = obj_dfint.drop(["fnlwgt"], axis = 1)
test1 = test1.drop(["capital_gain"], axis = 1)
test1 = test1.drop(["capital_loss"], axis = 1)

(Xt1,Xscale) = normaliseData(obj_dfint.fnlwgt)
(Xt2,Xscale) = normaliseData(obj_dfint.capital_gain)
(Xt3,Xscale) = normaliseData(obj_dfint.capital_loss)

#combining category and integer columns
frames = [test1, Xt1, Xt2, Xt3, obj_df]
Data1 = pd.concat(frames, axis = 1)

#splitting a dataframe to form input 
data1 = Data1.iloc[:,:14]
data1
#removal of column based on correlation matrix
corr = Data1.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

data1 = data1.drop(["education"], axis = 1)
data1 = data1.drop(["native_country"], axis = 1)
data1 = data1.drop(["race"], axis = 1)

#inputs
dataip = pd.DataFrame(data1)
#outputs
dataop = Data1.iloc[:,14:]
#Uncomment this line for data size = 16k
X_train1, X_test1, y_train1, y_test1 = train_test_split(dataip, dataop, train_size=0.8)

#executing various models for different values of input
for j in range(1, 5, 1):
   f1_acc = [] 
   count = []
   for i in range(1, 20, 1):
      a = i / 20.0
      X_train, X_test, y_train, y_test = train_test_split(X_train1, y_train1, train_size=a)
      
      n=len(X_train)
#logistic regression
      if j == 1:
         save = 'Logistic.png'
         model = LogisticRegression(intercept_scaling=2, max_iter=1000)
         model.fit(X_train, y_train)

#Decision Tree
      if j == 2:
         model = DecisionTreeClassifier(criterion='gini', min_samples_split=8)
         model.fit(X_train, y_train)
         save = 'DT.png'

#random forest
      if j == 3:
         model = RandomForestClassifier(criterion='gini', min_samples_split=8)
         model.fit(X_train, y_train)
         save = 'RF.png'
         
#Support Vector Machines          
      if j == 4:       
        model = SVC(kernel='rbf')  
        model.fit(X_train, y_train) 
        save = 'SVM.png'

# make predictions
      expected = y_test1
      predicted = model.predict(X_test1)
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
      



  
