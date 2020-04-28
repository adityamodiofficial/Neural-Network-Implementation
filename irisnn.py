# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 19:50:23 2019

@author: sandhu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.utils import shuffle
import random
import math
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
df=pd.read_csv("iris.csv")
df.info()
df=shuffle(df)
print(df.head())
print(df["class"].value_counts())
df["type"]=(df["class"]=="Iris-virginica").astype(np.int)
print(df.head())
train=df.sample(frac=.7,random_state=42)
test=df.drop(train.index)
print(len(train),len(test))
trainx=train.drop(["class","type"],axis=1)
trainy=train["type"]
print(trainx.head())
print(trainy.head())
X_train=np.transpose(trainx.as_matrix())
Y_train=np.transpose(trainy.as_matrix())
print(X_train.shape,Y_train.shape)
d=np.random.random((3,4))
w1=d
print(w1)
d=np.random.random((3,1))
b1=d
d=np.random.random((1,3))
w2=d
d=np.random.random((1,1))
b2=d
iterations=[]
loss=[]
for i in range(10000):
    z1=np.dot(w1,X_train)+b1
    #print(z1.shape)
    a1=1/(1+np.exp(-(z1)))
    z2=np.dot(w2,a1)+b2
    a2=1/(1+np.exp(-(z2)))
    #print(a2.shape,Y_train.shape)
    dz2=a2-Y_train
    #print(dz2.shape)
    dw2=(np.dot(dz2,z1.T))/(len(train)) #inplace z1 try using a1 the corret one 
    db2=(np.sum(dz2,axis=1,keepdims=True))/(len(train))
    dz1=(np.dot(w2.T,dz2))*(a1*(1-a1))
    dw1=np.dot(dz1,X_train.T)/len(train)
    db1=(np.sum(dz1,axis=1,keepdims=True))/(len(train))
    w1=w1-0.01*dw1
    b1=b1-0.01*db1
    w2=w2-0.01*dw2
    b2=b2-0.01*db2
    if i == 4 or i==9999:
        print(w2)
    iterations.append(i)
    j=-(Y_train*np.log(a2)+(1.0001-Y_train)*(np.log(1.0001-a2)))
    j=np.sum(j)/(len(train))
    loss.append(j)
#sns.lineplot(x=iterations,y=loss)

testx=test.drop(["class","type"],axis=1)
testy=test["type"]
X_test=np.transpose(testx.as_matrix())
Y_test=np.transpose(testy.as_matrix())
#print(w1.shape,X_test.shape,b.shape)
print(w1.shape,w2.shape)
z1=np.dot(w1,testx.T)+b1
z2=np.dot(w2,1/(1+np.exp(-z1)))+b2   
aa=1/(1+np.exp(-(z2)))
prediction=np.round(aa)
print(prediction.shape,Y_test.shape)
print(prediction) 
print(Y_test)

cm=confusion_matrix(Y_test,prediction.T)
print(cm)
recall=cm[0][0]/(cm[0][0]+cm[1][0])
precision=cm[0][0]/(cm[0][0]+cm[0][1])
accuracy=(cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
print("Accuracy :: %.2f " % accuracy)
print("Recall :: %.2f " % recall)
print("Precision :: %.2f " % precision)
print("F1-Score :: %.2f " % (2*((precision*recall)/(precision+recall))))
auc = roc_auc_score(testy, prediction.T)
print('AUC :: %.2f' % auc)           
fpr, tpr, thresholds = roc_curve(testy, prediction.T)
plt.plot(fpr, tpr, color='red', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()




