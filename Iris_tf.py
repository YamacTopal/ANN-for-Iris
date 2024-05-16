# -*- coding: utf-8 -*-
"""
Created on Wed May  8 19:24:12 2024

@author: Yamaç
"""

from ucimlrepo import fetch_ucirepo 
import tensorflow
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score
import numpy

iris = fetch_ucirepo(id=53) 


X = iris.data.features 
y = iris.data.targets 

ct = ColumnTransformer(transformers=[('encoder', sklearn.preprocessing.OneHotEncoder(), [0])], remainder='passthrough')
y = numpy.array(ct.fit_transform(y))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

sc = sklearn.preprocessing.StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

ann = tensorflow.keras.models.Sequential()

ann.add(tensorflow.keras.layers.Dense(units=32, activation='relu'))

ann.add(tensorflow.keras.layers.Dense(units=64, activation='relu'))

ann.add(tensorflow.keras.layers.Dense(units=3, activation='softmax'))

ann.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics=['accuracy'])

ann.fit(X_train, y_train, batch_size=32, epochs=100)

y_pred = ann.predict(X_test)

    

for i in range(len(y_pred)):
    for j in range(0, 3):
        if y_pred[i,j] >= 0.5:
            y_pred[i, j] = 1
        else:
            y_pred[i, j] = 0
        



accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

f1 = f1_score(y_test, y_pred, average='macro')  # Macro-averaging
print("F1-score (Macro):", f1)
