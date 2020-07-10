# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 21:39:54 2020

@author: gabri
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

data = pd.read_csv('data/iris.csv')

X= data.iloc[:,:-1]#.values
Y=data['class']#.values

from sklearn.preprocessing import LabelEncoder
label_encoder= LabelEncoder()
Y=label_encoder.fit_transform(Y)
Y_dummy= np_utils.to_categorical(Y)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,Y_dummy,train_size=0.75)


classificador = Sequential()


classificador.add(Dense(units=4, activation='relu',input_dim=4))
# quantidade de neurônios = (atributos+qtdadae de saidas)/2
classificador.add(Dense(units=4, activation='relu'))
classificador.add(Dense(units=3, activation='softmax'))
classificador.compile(optimizer='adam',loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])

classificador.fit(x=x_train,y=y_train,batch_size=10,epochs=1000 )

#teste automático
resultado= classificador.evaluate(x=x_test,y=y_test)

#teste """manual"""
previsoes= classificador.predict(x=x_test)
previsoes=previsoes>0.5
import numpy as np
y_test_index=[np.argmax(i) for i in y_test]
previsoes_index=[np.argmax(i) for i in previsoes]

from sklearn.metrics import confusion_matrix

matrix=confusion_matrix(previsoes_index,y_test_index)


