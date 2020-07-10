# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 21:39:54 2020

@author: gabri
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

data = pd.read_csv('data/iris.csv')

X= data.iloc[:,:-1]#.values
Y=data['class']#.values

from sklearn.preprocessing import LabelEncoder
label_encoder= LabelEncoder()
Y=label_encoder.fit_transform(Y)
Y_dummy= np_utils.to_categorical(Y)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,Y_dummy,train_size=0.75)


def generate_network():
    
    network = Sequential()
    network.add(Dense(units=4, activation='relu',input_dim=4))
    # quantidade de neurônios = (atributos+qtdadae de saidas)/2
    network.add(Dense(units=4, activation='relu'))
    network.add(Dense(units=3, activation='softmax'))
    network.compile(optimizer='adam',loss='categorical_crossentropy',
                          metrics=['categorical_accuracy'])
    return classificador

network= KerasClassifier(build_fn=generate_network,
                         batch_size=10,
                         epochs=100)

resultados= cross_val_score(estimator=network,
                            X=X, y=Y,
                            cv=10, scoring='accuracy')

media_performance_modelo=resultados.mean()
std=resultados.std()
