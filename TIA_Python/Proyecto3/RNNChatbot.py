# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 23:06:14 2019

@author: ASUS
"""
import os 
import json 
import nltk
import gensim 
import numpy as np
from gensim import corpora, models, similarities
import pickle
os.chdir("D:/Dave/code/2019_02/TIA_Python/Proyecto3")

# ------------------- MAIN PARAMETERS --------------------------------------
WORD2VEC_DICTIONARY = 'apnews_sg/word2vec.bin'
NAME_CONVERSATION = '/conversation.json'
DUMP_FILE = 'processed_conversation.pickle'
#--------------------------------------------------------------------------- 





model = gensim.models.Word2Vec.load(WORD2VEC_DICTIONARY) #Caragando Word2Vec database
path2 = "corpus" #Caragando archivos con conversaciones.

file = open(path2 + NAME_CONVERSATION)
data = json.load(file)
cor = data["conversations"] # Diccionario con lista de tamaño 13

x = []
y = []


for i in range(len(cor)):
    for j in range(len(cor[i])):
        if j < len(cor[i]) - 1:
            x.append(cor[i][j])
            y.append(cor[i][j+1])
            
tok_x = []
tok_y = []


#print("x\n", len(x), "\n", x)
#print("y\n", len(y), "\n", y)

# Tokenizando y dando formato a las palabras 
for i in range(len(x)):
    tok_x.append(nltk.word_tokenize(x[i].lower()))
    tok_y.append(nltk.word_tokenize(y[i].lower()))


print("W2V size: " , len(model.wv.vocab))
# Creando matriz base de 300 para one hot encoding
sentend = np.ones((300, ), dtype=np.float32)


vec_x = []
for sent in tok_x:
    sentvec = [model[w] for w in sent if w in model.wv.vocab]
    vec_x.append(sentvec)

vec_y = []
for sent in tok_y:
    sentvec = [model[w] for w in sent if w in model.wv.vocab]
    vec_y.append(sentvec)

print("After vocabulary: Vec_x \n", len(vec_x), "-", len(vec_x[0]))
print("After vocabulary: Vec_y \n", len(vec_y), "-", len(vec_y[0]))


#len(vec_x) = 86
#len(vec_x[0]) = 7
#len(vec_x[0][0]) = 300
# Adecuando a arquitectura d RNN
for tok_sent in vec_x:
    tok_sent[14:] = [] # Vec_x es una matriz de 86x7x300
    tok_sent.append(sentend)
#len(vec_x) = 86
#len(vec_x[0]) = 8
#len(vec_x[0][0]) = 300
for tok_sent in vec_x:
    if len(tok_sent)<15:
        for i in range(15-len(tok_sent)):
            tok_sent.append(sentend)  
#len(vec_x) = 86
#len(vec_x[0]) = 8
#len(vec_x[0][0]) = 300


#len(vec_y) = 86
#len(vec_y[0]) = 9
#len(vec_y[0][0]) = 300
for tok_sent in vec_y:
    tok_sent[14:]=[]
    tok_sent.append(sentend)    
#len(vec_y) = 86
#len(vec_y[0]) = 10
#len(vec_y[0][0]) = 300
for tok_sent in vec_y:
    if len(tok_sent)<15:
        for i in range(15-len(tok_sent)):
            tok_sent.append(sentend)
#len(vec_y) = 86
#len(vec_y[0]) = 15
#len(vec_y[0][0]) = 300

# Saving preprocesed data in a dump file
with open('conversation.pickle','wb') as f:
    pickle.dump([vec_x,vec_y],f)         
    
    
    
    

            


