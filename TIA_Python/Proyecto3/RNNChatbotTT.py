# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 01:05:53 2019

@author: ASUS
"""
import os 
import pickle
import numpy as np
import gensim
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dropout
from tensorflow.keras.losses import CosineSimilarity
import theano
theano.config.optimizer="None"

# ------------------- MAIN PARAMETERS --------------------------------------
DUMP_FILE = 'processed_conversation.pickle'
KNOWLEDGE_FILE = 'entrenamiento_1000e.h5'
epocas = 1000
#--------------------------------------------------------------------------- 


#Cargando archivo dump de segmento previo
with open(DUMP_FILE, 'rb') as f:
    vec_x,vec_y=pickle.load(f) 

# Dando formato bajo numpy
vec_x = np.asarray(vec_x, dtype=np.float64)
vec_y = np.asarray(vec_y, dtype=np.float64)
print("X shape: ", vec_x.shape)
print("Y shape: ", vec_y.shape)

# Obteniendo conjunto de entrenamiento y conjunto de salida
x_train,x_test, y_train,y_test = train_test_split(vec_x, vec_y, test_size=0.2, random_state=1)
print("Conjunto de entrenamiento: ", x_train.shape, x_test.shape)
print("Conjunto de prueba: ", y_train.shape, y_test.shape)


model = Sequential()
model.add(LSTM(300, return_sequences = True))
model.add(LSTM(300, return_sequences = True))
model.add(LSTM(300, return_sequences = True))
model.add(LSTM(300, return_sequences = True))
model.add(Dropout(0.5))
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
model.compile(loss='cosine_similarity', optimizer='adam', metrics=['categorical_accuracy'])
model.fit(x_train, y_train, nb_epoch=epocas,validation_data=(x_test, y_test))
model.save(KNOWLEDGE_FILE);

