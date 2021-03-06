#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 16:41:34 2018

@author: hamid
"""

import scipy.io

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

file = 'bank-full.xlsx'

data = pd.read_excel(file, sheet_name = 'bank-full')

labelencoder_X = LabelEncoder()
data['job'] = labelencoder_X.fit_transform(data['job'])
data['marital'] = labelencoder_X.fit_transform(data['marital']) 
data['education'] = labelencoder_X.fit_transform(data['education']) 
data['default'] = labelencoder_X.fit_transform(data['default']) 
data['housing'] = labelencoder_X.fit_transform(data['housing']) 
data['loan'] = labelencoder_X.fit_transform(data['loan']) 
data['contact'] = labelencoder_X.fit_transform(data['contact']) 
data['month'] = labelencoder_X.fit_transform(data['month']) 
data['poutcome'] = labelencoder_X.fit_transform(data['poutcome']) 
data['y'] = labelencoder_X.fit_transform(data['y']) 

data = np.array(data)
#np.random.shuffle(data) # 1 random

b = -1  # which columns should be feeded in to the model
average_method = 'weighted'  # None, 'micro', 'macro', 'weighted'
f1_acc = []; nl_acc = []

X = data[:,0:b]
Y = data[:,-1]
    
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#c = np.column_stack((x_train,y_train))
#np.random.shuffle(c)
#x_train = c[:,0:-1]
#y_train = c[:,-1]

target = [0.02,10,20,30,40,50,60,70,80,90,100] #range(100,x_train.shape[0],1000)

for i in target:
    
    a = int(np.round(i * x_train.shape[0] / 100))
    
    x_train1 = x_train[0:a,:]
    y_train1 = y_train[0:a]
    
    model = LogisticRegression()
    #model = svm.SVC(kernel='linear',verbose=1)
    #model = LinearSVC(verbose=1) #,max_iter=10000)
    #model = KNeighborsClassifier(n_neighbors = 4)
    #model = DecisionTreeClassifier()
    #model = RandomForestClassifier()
    #model = GaussianNB()
    
    model.fit(x_train1, y_train1)
    
    #print(model.coef_)

    #print(model.score(x_test,y_test))

    ######### Cross Validation
    #ss = ShuffleSplit(n_splits = 20, test_size = 0.5) # number of datasets
    #scores=cross_val_score(model, X, Y, cv = ss, scoring = 'f1_macro')
    #print("Cross-Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    #predictions_LR = model.predict(data[:,0:b])
    predictions_LR = model.predict(x_test)
    
    #print('F1 Score (weighted) = ',  f1_score(data[:,-1], predictions_LR, average = average_method))
    #print('Accuracy Score = ', accuracy_score(data[:,-1], predictions_LR))
    #f1_acc.append(f1_score(data[:,-1], predictions_LR, average = average_method))
    #nl_acc.append(accuracy_score(data[:,-1], predictions_LR))
    
    #print(f1_score(y_test, predictions_LR, average = average_method))
    f1_acc.append(f1_score(y_test, predictions_LR, average = average_method))
    nl_acc.append(accuracy_score(y_test, predictions_LR))
    
f1_acc = np.array(f1_acc).ravel()
nl_acc = np.array(nl_acc).ravel()
target = np.array(target).ravel()
result = np.column_stack((target,f1_acc))
result = np.column_stack((result,nl_acc))

#fig, ax = plt.subplots(figsize=(12,8))
#ax.scatter(target, f1_acc, c = 'b', marker = 'o', label = 'F1 Accuracy')
#ax.scatter(target, nl_acc, c = 'r', marker = 'x', label = 'Normal Accurcay')
#ax.legend()
#ax.set_xlabel('Dataset Size')
#ax.set_ylabel('Accuracy')
#fig.savefig('GaNB.eps')

#scipy.io.savemat('result_' + 'LR' + '.mat', {'result': result})