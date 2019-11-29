# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 01:29:41 2019
@author: ASUS
"""
import os
import warnings
from scipy import spatial
import numpy as np
import gensim
import nltk
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import models
from tensorflow.keras.models import load_model
import theano
theano.config.optmizer="None"
warnings.filterwarnings('ignore')

# ------------------- MAIN PARAMETERS --------------------------------------
KNOWLEDGE_FILE = 'entrenamiento_1000e.h5'
WORD2VEC_DICTIONARY = 'apnews_sg/word2vec.bin'
#--------------------------------------------------------------------------- 

model = load_model(KNOWLEDGE_FILE)
mod = gensim.models.Word2Vec.load(WORD2VEC_DICTIONARY)


def optimize_answer(answer):
    for word in range(len(answer)-1):
        if answer[word] == answer[word+1]:
            answer = answer[:word+1]
            return answer

def optime_answer_v2(answer):
    answer = answer.replace('kleiser', '')
    answer = answer.replace('karluah', '')
    answer = answer.replace('ballets', '')
    return answer
    
to_ask = 30
print('Escriba <quit> para finalizar la conversacion')
for asked_questions in range(to_ask):
    x = input("Di algo: ")
    if x == 'quit':
        break
    
    # Dando formato de entrenamiento
    sentend = np.ones((300,), dtype=np.float32)
    sent = nltk.word_tokenize(x.lower())
    sentvec = [mod[w] for w in sent if w in mod.wv.vocab]
    sentvec[14:] = []
    sentvec.append(sentend)
    if len(sentvec) < 15:
        for i in range(15 - len(sentvec)):
            sentvec.append(sentend)        
    sentvec=np.array([sentvec])
    
    predictions = model.predict(sentvec)
    outputlist= [mod.most_similar([predictions[0][i]])[0][0] for i in range(15)]
    #outputlist= optimize_answer(outputlist)
    output = ' '.join(outputlist)
    output = optime_answer_v2(output)
    
    print('Dave dice: ', output) 