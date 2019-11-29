# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 23:58:36 2019

@author: ASUS
"""
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


import numpy
import tflearn
import tensorflow
import random

training = [45, 63]
output = [45, 6]

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, training[1] ])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, output[1], activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
model.load('save_network/model.tfl')

print('El modelo se ha cargado exitosamente')

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


def chat():
    print("El bot te esta esperando... ")
    while True:
        inp = input("You: ")
        if inp.lower() == "salir":
            print("Bien, regreso a dormir...")
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))

if __name__ == '__main__':
    chat()