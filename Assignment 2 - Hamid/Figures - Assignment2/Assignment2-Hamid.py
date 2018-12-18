#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 11:10:18 2018

@author: hamid
"""

import pandas as pd
import numpy as np
import scipy.io
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
import warnings
warnings.filterwarnings('ignore')

#file = 'bank-additional-full-1.xlsx'
#data = pd.read_excel(file, sheet_name = 'bank-additional-full')

#labelencoder = LabelEncoder()

#data['job'] = labelencoder.fit_transform(data['job'])
#data['marital'] = labelencoder.fit_transform(data['marital']) 
#data['education'] = labelencoder.fit_transform(data['education']) 
#data['default'] = labelencoder.fit_transform(data['default']) 
#data['housing'] = labelencoder.fit_transform(data['housing']) 
#data['loan'] = labelencoder.fit_transform(data['loan']) 
#data['contact'] = labelencoder.fit_transform(data['contact']) 
#data['month'] = labelencoder.fit_transform(data['month'])
#data['day_of_week'] = labelencoder.fit_transform(data['day_of_week']) 
#data['poutcome'] = labelencoder.fit_transform(data['poutcome'])
#data['y'] = labelencoder.fit_transform(data['y'])   # 1 should be detected

#data = np.array(data)
#scipy.io.savemat('encode_data_bank_1.mat', {'data': data})

X = []; Y = [];
temp = []; temp = scipy.io.loadmat('X.mat')
X = temp['X']
temp = []; temp = scipy.io.loadmat('Y.mat')
Y = temp['Y']

#X = X[0:45211,0:10]
#Y = Y[0:45211,:]

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

#from sklearn.feature_selection import RFE
#LR = LogisticRegression()
#rfe = RFE(estimator=LR, n_features_to_select=10, step=1)
#rfe.fit(X, Y)
#rfe.ranking_ # 1 is the best

#import seaborn as sns
#sns.set()
#ax = sns.heatmap(X)

#print(len(np.unique(data[:,0])),np.min(data[:,0]),np.max(data[:,0]),np.mean(data[:,0]),np.std(data[:,0]))
#data[:,0] = data[:,0] - np.mod(data[:,0],10)

average_method = 'weighted'  # None, 'micro', 'macro', 'weighted'
f1_acc = []; nl_acc = []

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#c = np.column_stack((x_train,y_train))
#np.random.shuffle(c)
#x_train = c[:,0:-1]
#y_train = c[:,-1]

target = [0.008,10,20,30,40,50,60,70,80,90,100]

for i in target:
    
    a = int(np.round(i * x_train.shape[0] / 100))
    
    x_train_target = x_train[0:a,:]
    y_train_target = y_train[0:a]
    
    model = LogisticRegression()
    #model = svm.SVC(kernel='linear')
    #model = LinearSVC(verbose=1) #,max_iter=10000)
    #model = KNeighborsClassifier(n_neighbors = 2)
    #model = DecisionTreeClassifier()
    #model = RandomForestClassifier()
    #model = GaussianNB()
    
    model.fit(x_train_target, y_train_target)
    
    #print(model.coef_)

    #print(model.score(x_test,y_test))

    ######### Cross Validation    
    #predictions_LR = model.predict(data[:,0:b])
    predictions_LR = model.predict(x_test)
    
    #print('F1 Score (weighted) = ',  f1_score(data[:,-1], predictions_LR, average = average_method))
    #print('Accuracy Score = ', accuracy_score(data[:,-1], predictions_LR))
    #f1_acc.append(f1_score(data[:,-1], predictions_LR, average = average_method))
    #nl_acc.append(accuracy_score(data[:,-1], predictions_LR))
    
    #print(f1_score(y_test, predictions_LR, average = average_method))
    
    #X_m = np.row_stack((x_train_target,x_test))
    #Y_m = np.row_stack((y_train_target,y_test))
    X_m = x_test
    Y_m = y_test

    ss = ShuffleSplit(n_splits = 20, test_size = 0.5) # number of datasets
    temp = [];
    for train_index, test_index in ss.split(X_m):
        model = []; model = LogisticRegression()
        model.fit(X_m[train_index], Y_m[train_index])
        temp.append(f1_score(Y_m[test_index], model.predict(X_m[test_index]), average = average_method)) #pos_label = 1))
        #temp.append(model.score(X_m[test_index],Y_m[test_index]))
    print("F1 Score: %0.3f (+/- %0.3f)" % (np.mean(temp), np.std(temp)), f1_score(y_test, predictions_LR, average = average_method))
    
    f1_acc.append([f1_score(y_test, predictions_LR, average = average_method), np.std(temp)])
    #f1_acc.append([f1_score(y_test, predictions_LR, average = average_method))
    #nl_acc.append(accuracy_score(y_test, predictions_LR))
    
#f1_acc = np.array(f1_acc).ravel()
#nl_acc = np.array(nl_acc).ravel()
#target = np.array(target).ravel()
result = np.column_stack((target,f1_acc))
#result = np.column_stack((result,nl_acc))

#fig, ax = plt.subplots(figsize=(12,8))
#ax.scatter(target, f1_acc, c = 'b', marker = 'o', label = 'F1 Accuracy')
#ax.scatter(target, nl_acc, c = 'r', marker = 'x', label = 'Normal Accurcay')
#ax.legend()
#ax.set_xlabel('Dataset Size')
#ax.set_ylabel('Accuracy')
#fig.savefig('GaNB.eps')

#scipy.io.savemat('result_' + 'GNB' + '.mat', {'result': result})