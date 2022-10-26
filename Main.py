# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 22:30:13 2022

@author: USER
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score

import os

path = os.listdir('brain_tumor/Training/')
classes = {'no_tumor':0, 'pituitary_tumor':1}

import cv2
X = []
Y = []
for cls in classes:
    pth = 'brain_tumor/Training/'+cls
    for j in os.listdir(pth):
        img = cv2.imread(pth+'/'+j, 0)
        img = cv2.resize(img, (200,200))
        X.append(img)
        Y.append(classes[cls])
X = np.array(X)
Y = np.array(Y)

X_updated = X.reshape(len(X), -1)

np.unique(Y)

pd.Series(Y).value_counts()

X.shape, X_updated.shape

#cv2.imshow('Image',X[0])


plt.imshow(X[0], cmap='gray')

X_updated = X.reshape(len(X), -1)
X_updated.shape

xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10,
                                               test_size=.20)

xtrain.shape, xtest.shape

print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())
xtrain = xtrain/255
xtest = xtest/255
print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())


from sklearn.decomposition import PCA

print(xtrain.shape, xtest.shape)

pca = PCA(.98)
pca_train = pca.fit_transform(xtrain)
pca_test = pca.transform(xtest)
#pca_train = xtrain
#pca_test = xtest


#from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')

#lg = LogisticRegression(C=0.1)
#lg.fit(xtrain, ytrain)
sv = SVC()
sv.fit(xtrain, ytrain)
y_pred=sv.predict(xtest)
'''
from sklearn.model_selection import GridSearchCV
parameters={'C':[0.01,0.05,0.1,0.5,1],
            'degree':[2,3,4,5],
            'gamma':[0.001,0.01,0.1,0.5,1],
            'kernel':['rbf','poly']
    }

sv = SVC()
grid=GridSearchCV(sv, parameters, verbose=2, scoring="accuracy")
sv.fit(xtrain, ytrain)
#y_pred=sv.predict(xtest)
print(grid.best_params_)
print(grid.best_estimator_)
'''
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(ytest,y_pred))

#print("Training Score:", lg.score(xtrain, ytrain))
#print("Testing Score:", lg.score(xtest, ytest))

print("Training Score:", sv.score(xtrain, ytrain))
print("Testing Score:", sv.score(xtest, ytest))

from sklearn.metrics import plot_confusion_matrix
clf = SVC(random_state=10)
clf.fit(xtrain, ytrain)
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(ytest, y_pred)

import seaborn as sn
sn.set(font_scale=1.5)
sn.heatmap(confusion, square=True, annot=True, fmt='g', annot_kws={"size":16})
plt.title("Confusion Matrix")
plt.xlabel('Predicted Label')
plt.ylabel('True label');
plt.show()

pred = sv.predict(xtest)

misclassified=np.where(ytest!=pred)
misclassified

print("Total Misclassified Samples: ",len(misclassified[0]))
print(pred[36],ytest[36])




dec = {0:'No Tumor', 1:' Tumor Detected'}
plt.figure(figsize=(12,8))
p = os.listdir('brain_tumor/Testing/')
c=1
for i in os.listdir('brain_tumor/Testing/no_tumor/')[:9]:
    plt.subplot(3,3,c)
    
    img = cv2.imread('brain_tumor/Testing/no_tumor/'+i,0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c+=1
    
    
plt.figure(figsize=(12,8))
p = os.listdir('brain_tumor/Testing/')
c=1
for i in os.listdir('brain_tumor/Testing/pituitary_tumor/')[:16]:
    plt.subplot(4,4,c)
    
    img = cv2.imread('brain_tumor/Testing/pituitary_tumor/'+i,0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c+=1