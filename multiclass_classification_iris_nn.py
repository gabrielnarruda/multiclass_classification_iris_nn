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

#from sklearn.preprocessing import LabelEncoder
#label_encoder= LabelEncoder()
#Y=label_encoder.fit_transform(Y)
#Y_dummy= np_utils.to_categorical(Y)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,Y,train_size=0.75)


classificador = Sequential()


classificador.add(Dense(units=4, activation='relu',input_dim=4))
# quantidade de neurônios = (atributos+qtdadae de saidas)/2
classificador.add(Dense(units=4, activation='relu'))
classificador.add(Dense(units=3, activation='softmax'))
classificador.compile(optimizer='adam',loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])

classificador.fit(x=x_train,y=y_train,batch_size=10,epochs=1000 )

### OBS::: Ao ajustar o modelo aos dados foi encontrado o seguinte erro:
#ValueError: Error when checking target: expected dense_13 to have shape (3,) but got array with shape (1,)
#Este erro acontece pois o espaço vetorial de Y deve ser igual à quantidade de outputs ad rede, dessa forma, 
# acda valor do array Y deve descrever numéricamente  sua relação com todas as classes. Segue exemplo.
# iris setosa     1 0 0
# iris virginica  0 1 0
# iris versicolor 0 0 1