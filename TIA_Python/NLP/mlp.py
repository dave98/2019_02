import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt
from sklearn import svm

class mlp:
    def __init__(self, n_entry_layer, n_hidden_layer, n_per_hidden_layer, n_output_layer, activation_function='none'):
        self.n_entry_layer = int(n_entry_layer + 1) # Numero de neuronas capa de entrada // +1 por Bias
        self.n_hidden_layer = int(n_hidden_layer) # Numero de capas ocultas
        self.n_per_hidden_layer = int(n_per_hidden_layer + 1) # Numero de neuronas en capas ocultas // +1 por Bias
        self.n_output_layer = int(n_output_layer) # Numero de neuronas capa de salida
        self.activation_function = activation_function

        # -------- Temporal Containers ------------
        self.result_y_data = None
        self.last_errors = np.zeros((self.n_output_layer), dtype='float64')

        #--------- NEURONAS ----------------
        self.entry_layer = np.zeros((self.n_entry_layer), dtype='float64') # nx1 (verticalx1)
        self.hidden_layer = np.zeros((self.n_hidden_layer, self.n_per_hidden_layer - 1), dtype='float64') # nxm -> n:hidden_layer, m:n_per_hidden_layer
        self.hidden_layer = np.c_[np.ones(self.hidden_layer.shape[0]), self.hidden_layer] # Añadiendo BIAS
        self.output_layer = np.zeros((self.n_output_layer), dtype='float64')

        #--------- PESOS -------------------
        #print( type(self.n_per_hidden_layer) )
        if self.n_hidden_layer == 0:
            # nxm -> n(numero de neuronas en capa de salida(no existe bias)) m(numero de neuronas en capa de entrada considerando bias)
            self.entry_weights = np.random.rand(self.n_output_layer, self.n_entry_layer) # Conexión directa entre entrada y salida
        else:
            # nxm -> n(numero de neuronas en la capa oculta sin contar el bias(no posee peso)) m(numero de neuronas en la capa inicial mas el bias)
            self.entry_weights = np.random.rand(self.n_per_hidden_layer-1, self.n_entry_layer) # Conexión entre entrada y primera capa oculta
            # nxmxo -> n(numero de capas ocultas - 1 descontado de la primera capa) m(numero de neuronas en capa objetivo sin contar el bias) o(numero de neuronas de capa fuente considerando bias)
            self.hidden_weigths = np.random.rand(self.n_hidden_layer-1, self.n_per_hidden_layer-1, self.n_per_hidden_layer) # Conexión entre capas ocultas, considerando configuración Bias
            # nxm -> n(numero de neuronas en capa de salida (no existe bias)) m(numero de neuronas en capa intermedia considerando bias)
            self.output_weights = np.random.rand(self.n_output_layer, self.n_per_hidden_layer) # Conexión entre hidden y output

    def reset(self):
        self.entry_layer = np.zeros((self.n_entry_layer), dtype='float64') # nx1 (verticalx1)
        self.hidden_layer = np.zeros((self.n_hidden_layer, self.n_per_hidden_layer - 1), dtype='float64') # nxm -> n:hidden_layer, m:n_per_hidden_layer
        self.hidden_layer = np.c_[np.ones(self.hidden_layer.shape[0]), self.hidden_layer] # Añadiendo BIAS
        self.output_layer = np.zeros((self.n_output_layer), dtype='float64')

        #--------- PESOS -------------------
        #print( type(self.n_per_hidden_layer) )
        if self.n_hidden_layer == 0:
            # nxm -> n(numero de neuronas en capa de salida(no existe bias)) m(numero de neuronas en capa de entrada considerando bias)
            self.entry_weights = np.random.rand(self.n_output_layer, self.n_entry_layer) # Conexión directa entre entrada y salida
        else:
            # nxm -> n(numero de neuronas en la capa oculta sin contar el bias(no posee peso)) m(numero de neuronas en la capa inicial mas el bias)
            self.entry_weights = np.random.rand(self.n_per_hidden_layer-1, self.n_entry_layer) # Conexión entre entrada y primera capa oculta
            # nxmxo -> n(numero de capas ocultas - 1 descontado de la primera capa) m(numero de neuronas en capa objetivo sin contar el bias) o(numero de neuronas de capa fuente considerando bias)
            self.hidden_weigths = np.random.rand(self.n_hidden_layer-1, self.n_per_hidden_layer-1, self.n_per_hidden_layer) # Conexión entre capas ocultas, considerando configuración Bias
            # nxm -> n(numero de neuronas en capa de salida (no existe bias)) m(numero de neuronas en capa intermedia considerando bias)
            self.output_weights = np.random.rand(self.n_output_layer, self.n_per_hidden_layer) # Conexión entre hidden y output

    def reader(self, route):
        data = pd.read_csv('Data/' + route + '.csv')
        data = data.to_numpy()
        return data

    def normalize(self, in_data):
        x_data = in_data[:, :in_data.shape[1]-1]
        y_data = in_data[:, [in_data.shape[1]-1]]
        x_data = x_data.astype('float64')

        media = np.mean(x_data, axis = 0)
        sdev = np.std(x_data, axis  = 0)

        x_data = x_data - media
        x_data = np.divide(x_data, sdev)
        x_data = np.concatenate((x_data, y_data), axis = 1)
        x_data = np.c_[np.ones(x_data.shape[0]), x_data]
        return x_data # Raw data, preprocesed

    def get_x_y_data(self, in_data):
        x_data = in_data[:, :in_data.shape[1]-1]
        y_data = in_data[:, [in_data.shape[1]-1]]
        return x_data, y_data # (n, m), (n*1)[Doble array]

    def apply_activation_function(self, layer_to_activate):
        if self.activation_function == 'none':
            return layer_to_activate
        elif self.activation_function == 'sigmoidal':
            return np.divide(1, 1 + np.exp(-1 * layer_to_activate) )
        else:
            raise Exception('Error en función de activacion')

    def calculate_error(self, y_data):
        return np.sum(np.divide( np.power((y_data - self.result_y_data), 2), 2), axis=0)

    def single_calculate_error(self, y_data):
        self.last_errors = np.divide( np.power(y_data - self.output_layer, 2), 2)

    # Para todo el grupo x_data NO UTIL
    def forward_operation(self, x_data, y_data): # La información de entrada ya incluye el bias en el primer elemento,
        self.result_y_data = np.zeros((y_data.shape[0], y_data.shape[1]), dtype='float64')
        if x_data.shape[1] == self.n_entry_layer:
            for i in range(0, x_data.shape[0]):
                if self.n_hidden_layer == 0: # Configuración entre capa de entrada y capa salida al no existir capa oculta.
                    # Forward operation sin hidden_layer
                    self.output_layer = self.apply_activation_function( np.sum( np.multiply(x_data[i], self.entry_weights), axis = 1 ) )
                else:
                    # Forward operation con hidden_layer
                    # Capa entrada  * Pesos entrada = Primera capa oculta
                    self.hidden_layer[0][1:self.hidden_layer.shape[1]] = self.apply_activation_function( np.sum( np.multiply(x_data[i], self.entry_weights), axis=1 ) ) # Multiplicamos entrada x por pesos entrada para sumar y añadir a la capa oculta
                    # A partir primera capa oculta * Pesos ocultos = Hasta ultima capa oculta
                    for j in range(0, self.n_hidden_layer-1): #No contamos una capa por ser parte de forward final
                        self.hidden_layer[j+1][1:self.hidden_layer.shape[1]] =  self.apply_activation_function( np.sum( np.multiply(self.hidden_layer[j], self.hidden_weigths[j]), axis = 1 ) )
                    # Ultima capa oculta * Pesos de salida = Output
                    self.output_layer =  self.apply_activation_function( np.sum( np.multiply(self.hidden_layer[self.hidden_layer.shape[0]-1], self.output_weights), axis = 1) )

                self.result_y_data[i] = self.output_layer
            return self.calculate_error(y_data) # Retornando error bajo la configuración actual
        else:
            raise Exception('Imposible realizar forward -> Inconsistencia entre datos de entrada y capa inicial')

    def single_forwar_operation(self, single_x_data, single_y_data):
        if self.n_hidden_layer == 0: # Configuración entre capa de entrada y capa salida al no existir capa oculta.
            self.output_layer = self.apply_activation_function( np.sum( np.multiply(single_x_data, self.entry_weights), axis = 1 ) )
        else:
            self.hidden_layer[0][1:self.hidden_layer.shape[1]] = self.apply_activation_function( np.sum( np.multiply(single_x_data, self.entry_weights), axis=1 ) ) # Multiplicamos entrada x por pesos entrada para sumar y añadir a la capa oculta
            for j in range(0, self.n_hidden_layer-1): #No contamos una capa por ser parte de forward final
                self.hidden_layer[j+1][1:self.hidden_layer.shape[1]] =  self.apply_activation_function( np.sum( np.multiply(self.hidden_layer[j], self.hidden_weigths[j]), axis = 1 ) )
            self.output_layer =  self.apply_activation_function( np.sum( np.multiply(self.hidden_layer[self.hidden_layer.shape[0]-1], self.output_weights), axis = 1) )
        self.single_calculate_error(single_y_data) # Calcula error y almacena en variable interna(last_errors) para ser usada en backward_operation

    def single_forward_evaluation(self, single_x_data, single_y_data):
        if self.n_hidden_layer == 0: # Configuración entre capa de entrada y capa salida al no existir capa oculta.
            self.output_layer = self.apply_activation_function( np.sum( np.multiply(single_x_data, self.entry_weights), axis = 1 ) )
        else:
            self.hidden_layer[0][1:self.hidden_layer.shape[1]] = self.apply_activation_function( np.sum( np.multiply(single_x_data, self.entry_weights), axis=1 ) ) # Multiplicamos entrada x por pesos entrada para sumar y añadir a la capa oculta
            for j in range(0, self.n_hidden_layer-1): #No contamos una capa por ser parte de forward final
                self.hidden_layer[j+1][1:self.hidden_layer.shape[1]] =  self.apply_activation_function( np.sum( np.multiply(self.hidden_layer[j], self.hidden_weigths[j]), axis = 1 ) )
            self.output_layer =  self.apply_activation_function( np.sum( np.multiply(self.hidden_layer[self.hidden_layer.shape[0]-1], self.output_weights), axis = 1) )
        #print(self.output_layer, '->', single_y_data)
        return self.output_layer

    def backward_operation(self, x_data, y_data, alpha_learning): # Single type
        # Iniciando con los pesos previos a la capa de salida.
        # Para cada peso previo a la capa de salida procedemos: &Etotal/&w1 = &Etotal/&salida1  *  &Salida1/&Red1 * &Red1/&w1
        # &Etotal/&salida1 = -(target1 - out1) ---- &Salida1/&Red1 = out1(1 - out1) ---- &Red1/&w1 = SalidaR1
        if self.n_hidden_layer == 0:
            print('Backward individual')
        else:
            f_output_container = self.output_weights.copy()
            f_hidden_container = self.hidden_weigths.copy()
            f_entry_container = self.entry_weights.copy()

            # Iniciamos con pesos de salida(output_weights) -> (Primera columna representa bias, no usar) Igualar considerando nuevos ese tamaño
            for i in range(0, self.output_weights.shape[0]): #Ultima capa de pesos
                new_output_weights = ( -1 * (y_data[i] - self.output_layer[i])) * (self.output_layer[i] * (1 - self.output_layer[i])) * self.hidden_layer[self.hidden_layer.shape[0]-1, 1:self.hidden_layer.shape[1]]
                f_output_container[i, 1:] = new_output_weights

            # Procedemos con pesos de capa oculta, esto no incluye a los pesos entre capa entrada y capa oculta
            hidden_start = (-1 * (y_data - self.output_layer)) * (self.output_layer * (1 - self.output_layer)) # De tamaño equivalente al ouput
            hidden_start = self.output_weights * hidden_start.reshape(hidden_start.shape[0], 1) # nxm (n:cantidad de salidas,  m: cantidad de pesos que conectan con la salidad, incluye bias)
            shaped_start = np.sum(hidden_start, axis = 0)

            temp_hidden_start = np.zeros((self.hidden_weigths.shape[1], self.hidden_weigths.shape[2]))
            for i in range(0, self.n_hidden_layer-1): #Navegando en capas desde final al principio
                i_back = self.hidden_weigths.shape[0]-i-1
                for j in range(1, self.hidden_weigths.shape[2]): # Sin considerar BIAS
                    new_hidden_weights = shaped_start[j] * (self.hidden_layer[i_back+1, 1:] * (1-self.hidden_layer[i_back+1, 1:])) * self.hidden_layer[i_back, 1:]
                    f_hidden_container[i, j-1, 1:] = new_hidden_weights
                    temp_hidden_start[j-1, 1:] = shaped_start[j] * (self.hidden_layer[i_back+1, 1:] * (1-self.hidden_layer[i_back+1, 1:])) * self.hidden_weigths[i_back, j-1, 1:]
                shaped_start = np.sum(temp_hidden_start, axis = 0)

            #Procedemos con los pesos de la capa de entrada, utilizando shaped_start como acumulado
            for i in range(0, self.entry_weights.shape[0]): # Sin considerar bias
                new_entry_weights = shaped_start[i+1] * (self.hidden_layer[0, i+1] * (1 - self.hidden_layer[0, i+1])) * x_data
                f_entry_container[i, 1:] = new_entry_weights[1:]

            #Aplicando Learning rate:
            f_output_container = alpha_learning * f_output_container
            f_hidden_container = alpha_learning * f_hidden_container
            f_entry_container = alpha_learning * f_entry_container

            self.output_weights[:,1:] = self.output_weights[:, 1:] - f_output_container[:, 1:]
            self.hidden_weigths[:,:,1:] = self.hidden_weigths[:, :, 1:] - f_hidden_container[:, :, 1:]
            self.entry_weights[:,1:] = self.entry_weights[:, 1:] - f_entry_container[:, 1:]

    def gradiente_descendiente(self, x_data, y_data, alpha_learning, ephoques): #Bloques x_data, y_data
        for i in range(0, ephoques):
            error_at_this_ephoque = 0.0
            for j in range(0, x_data.shape[0]):
                self.single_forwar_operation(x_data[j], y_data[j])
                error_at_this_ephoque += np.sum(self.last_errors)
                self.backward_operation(x_data[j], y_data[j], alpha_learning)

    def accuracy_lever(self, x_data, y_data):
        accerted_values = 0
        total_values = x_data.shape[0]
        for i in range(0, x_data.shape[0]):
            temp1 = self.single_forward_evaluation(x_data[i], y_data[i])
            temp1 = int(np.round_(temp1))
            if temp1 == y_data[i]:
                accerted_values += 1
        #print('Correcto en: ', accerted_values/total_values)
        return accerted_values/total_values

    def describe(self):
        print('Neuronas capa entrada: ', self.entry_layXer)
        print('Capas ocultas: ', self.n_hidden_layer)
        print('Neuronas por capa ocultas: ', self.n_per_hidden_layer)
        print('Neuronas capa salida', self.output_layer)
        print('Funcion de activacion: ', self.n_activation_function)

    def k_folds_cross_validation(self, n_folds, raw_data, alpha_learning, ephoques):
        k_folds = self.k_folds_function(n_folds, raw_data)
        prom_accuracy = 0
        for i in range(0, n_folds):
            temp_test_data = k_folds[i]
            temp_train_data = np.delete(k_folds, i, axis=0)
            temp_train_data = temp_train_data.reshape(-1, temp_test_data.shape[1])

            x_train_data, y_train_data = self.get_x_y_data(temp_train_data)
            x_test_data, y_test_data = self.get_x_y_data(temp_test_data)
            #self.reset()
            self.gradiente_descendiente(x_train_data, y_train_data, alpha_learning, ephoques)

            prom_accuracy += self.accuracy_lever(x_test_data, y_test_data)
        print('Ponderado: ', prom_accuracy/n_folds)

    def k_folds_function(self, n_folds, raw_data):
        category_1 = raw_data[raw_data[:, raw_data.shape[1]-1] == 1.]
        category_2 = raw_data[raw_data[:, raw_data.shape[1]-1] == 0.]

        category_1 = self.k_folds_formatter(n_folds, category_1)
        category_2 = self.k_folds_formatter(n_folds, category_2)

        category_1 = np.split(category_1, n_folds)
        category_2 = np.split(category_2, n_folds)
        category_1 = np.append(category_1, category_2, axis=1)
        return category_1

    def k_folds_formatter(self, n_folds, cat):
        if np.mod(cat.shape[0], n_folds) != 0:
            new_size = int(cat.shape[0]/ n_folds) * n_folds
            cat = cat[:new_size, :]
            return cat
        else:
            return cat

#----------------------------------- START HERE --------------------------------------
#np.random.seed(0) #Static random generator
mlp_1 = mlp(8, 2, 4, 1, activation_function='sigmoidal') # EntryLayer, HiddenLayer, NPerHiddenLayer, OutputLayer, Function
raw_data = mlp_1.reader('diabetes_2')
#raw_data = mlp_1.reader('heart_2')
raw_data = mlp_1.normalize(raw_data)
#print(raw_data.shape)
mlp_1.k_folds_cross_validation(3, raw_data, 0.5, 500)



#------------------------------------ SVM --------------------------------------------



# -------------------------------- DESCRIPCION --------------------------------------
# HIDDEN WEIGHTS # Varias matrices que representan la cantidad de capas ocultas menos uno (Entre entrada y primera capa oculta)
#  [[      ]   |
#   [      ]   | Numero de neuronas en la capa actual. Primera columna izquierda contiene los bias.
#   [      ]]  |
#  -- Cantidad de entrada de la capa previa (entrada u capa oculta anterior)
#
# OUPUT WEIGHTS # Estructura similar















"""
a = np.array([[1, 1, 1],
              [2, 2, 2],
              [3, 3, 3]])

#b = np.array([1, 2, 3])
b = np.array([2])
c = np.multiply(a, b)

print(a, '\n')
print(b.shape, '\n')
print(c)
"""

# -------------------------------- NOTAS NP ----------------------------------------
# Multiplicar con (np multiply):
# [1, 1, 1]               [1, 2, 3]
# [2, 2, 2] * [1, 2, 3] = [2, 4, 6]
# [3, 3, 3]               [3, 6, 9]
#
# [1, 1, 1]   [1, 1, 1]   [1, 1, 1]
# [2, 2, 2] * [2, 2, 2] = [4, 4, 4]
# [3, 3, 3]   [3, 3, 3]   [9, 9, 9]
#
# [1, 1, 1]   [[1]    [1, 1, 1]
# [2, 2, 2] *  [2] =  [4, 4, 4]
# [3, 3, 3]    [3]]   [9, 9, 9]
#
# [1, 1, 1]                  [2, 2, 2]
# [2, 2, 2] * [[2]] o [2] =  [4, 4, 4]
# [3, 3, 3]                  [6, 6, 6]























#####################################################################33
