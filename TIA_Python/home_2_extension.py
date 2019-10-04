from home_2 import data_container
from home_2 import reader_on
from home_2 import normalize
import matplotlib.pyplot as plt

import numpy as np

class one_vs_all:
    def __init__(self, in_data):
        taken_percentage = 0.95 #Porcentaje de los datos entrante usados para el entrenamiento
        taken_rows = int(in_data.shape[0]*taken_percentage) # Extrayendo del conjunto principal
        self.raw_data = in_data.copy()
        np.random.shuffle(in_data)

        self.train_set = in_data[:taken_rows-1, :]
        self.test_set = in_data[taken_rows-1:in_data.shape[0], :]

        self.special_atributes_from_last = np.unique(in_data.T[in_data.shape[1]-1])
        self.data_class_list = []

        for i in range(0, self.special_atributes_from_last.shape[0]):
            temp_data = np.c_[self.train_set[:, :self.train_set.shape[1]-1], np.zeros(self.train_set.shape[0])]
            temp_data[np.where(self.train_set == self.special_atributes_from_last[i])] = 1
            temp_data = temp_data.astype('float64')

            self.data_class_list.append(data_container(temp_data, taken_percentage = 0.95)) #Añadiendo los regresores

    def rebuild_clasifier(self, in_train_set, in_test_set):
        np.random.shuffle(in_train_set)
        np.random.shuffle(in_test_set)

        self.train_set = in_train_set
        self.test_set = in_test_set #Conservamos categorías de un primer momento

        self.data_class_list.clear()
        for i in range(0, self.special_atributes_from_last.shape[0]):
            temp_data = np.c_[self.train_set[:, :self.train_set.shape[1]-1], np.zeros(self.train_set.shape[0])]
            temp_data[np.where(self.train_set == self.special_atributes_from_last[i])] = 1
            temp_data = temp_data.astype('float64')
            self.data_class_list.append(data_container(temp_data, taken_percentage = 0.95)) #Añadiendo los regresores

    def learning_initialization(self, iteraciones=100, alpha_learning=0.01):
        for i in range(0, len(self.data_class_list)):
            self.data_class_list[i].gradiente_function_complete_vec(iteraciones, alpha_learning, is_shown='None')

    def predict_value(self, is_shown=False):
        war_test_set = self.test_set[:, :self.test_set.shape[1]-1]
        war_test_set = war_test_set.astype('float64')
        peace_test_set = self.test_set[:, [self.test_set.shape[1]-1]]

        predicted_evaluation = np.zeros(self.special_atributes_from_last.shape[0])
        accerted = 0
        for j in range(0, war_test_set.shape[0]):
            for i in range(0, len(self.data_class_list)):
                predicted_evaluation[i] = self.data_class_list[i].sigmoidal_function(war_test_set[j])
                #------------
                if is_shown:
                    print(i, '->', self.data_class_list[i].sigmoidal_function(war_test_set[j]))
            if is_shown:
                print(peace_test_set[j], ' -> ', np.where(self.special_atributes_from_last == peace_test_set[j]) , '\n')
            #--------------
            predicted_position_expected = int(np.where(self.special_atributes_from_last == peace_test_set[j])[0])
            if predicted_position_expected == np.argmax(predicted_evaluation):
                accerted+=1
        accerted = accerted * 100 / war_test_set.shape[0]
        print('Accuracy', accerted)
        return accerted

    def specific_k_folds_formatter(self, n_folds, cat):
        if np.mod(cat.shape[0], n_folds) != 0:
            new_size = int(cat.shape[0]/ n_folds) * n_folds
            cat = cat[:new_size, :]
            return cat
        else:
            return cat

    def specific_kfolds(self, n_folds):     #Just work for this dataset
        category_1 = self.raw_data[self.raw_data[:, self.raw_data.shape[1]-1] == self.special_atributes_from_last[0]]
        category_2 = self.raw_data[self.raw_data[:, self.raw_data.shape[1]-1] == self.special_atributes_from_last[1]]
        category_3 = self.raw_data[self.raw_data[:, self.raw_data.shape[1]-1] == self.special_atributes_from_last[2]]

        category_1 = self.specific_k_folds_formatter(n_folds, category_1)
        category_2 = self.specific_k_folds_formatter(n_folds, category_2)
        category_3 = self.specific_k_folds_formatter(n_folds, category_3)

        category_1 = np.split(category_1, n_folds)
        category_2 = np.split(category_2, n_folds)
        category_3 = np.split(category_3, n_folds)

        category_1 = np.append(category_1, category_2, axis=1)
        category_1 = np.append(category_1, category_3, axis=1)
        return category_1

    def specific_kfolds_cross_validation_dev(self, train_set, test_set, iteraciones, alpha_learning):
        self.rebuild_clasifier(train_set, test_set)
        self.learning_initialization()
        return self.predict_value()

    def specific_kfolds_cross_validation(self, n_folds=3, iteraciones=500, alpha_learning=0.001):
        k_folds = self.specific_kfolds(n_folds)
        media_accuracy = 0.0
        for i in range(0, n_folds):
            temp_test_data = k_folds[i] # Data Ready
            temp_train_data = np.delete(k_folds, i, axis=0)
            temp_train_data = temp_train_data.reshape(-1, temp_test_data.shape[1]) # Data Ready
            media_accuracy += self.specific_kfolds_cross_validation_dev(temp_train_data, temp_test_data, iteraciones, alpha_learning)
        print("Accuracy promedio: ", media_accuracy/n_folds)

class one_vs_one:
    def __init__(self, in_data):
        taken_percentage = 0.90 #Porcentaje de los datos entrante usados para el entrenamiento
        taken_rows = int(in_data.shape[0]*taken_percentage) # Extrayendo del conjunto principal
        self.raw_data = in_data.copy()
        np.random.shuffle(in_data)

        self.train_set = in_data[:taken_rows-1, :]
        self.test_set = in_data[taken_rows-1:in_data.shape[0], :]

        self.special_atributes_from_last = np.unique(in_data.T[in_data.shape[1]-1])
        self.data_class_list = []

        category_1 = self.train_set[self.train_set[:, self.train_set.shape[1]-1] == self.special_atributes_from_last[0]]
        category_2 = self.train_set[self.train_set[:, self.train_set.shape[1]-1] == self.special_atributes_from_last[1]]
        category_3 = self.train_set[self.train_set[:, self.train_set.shape[1]-1] == self.special_atributes_from_last[2]]
        self.train_set_atributes = np.array([category_1, category_2, category_3])

        for i in range(0, self.special_atributes_from_last.shape[0]):
            for j in range(i+1, self.special_atributes_from_last.shape[0]):
                #El indice i representará el valor 1 mientras j será 0
                temp_data = np.concatenate((self.train_set_atributes[i], self.train_set_atributes[j]), axis = 0)
                posiciones_a_cambiar = np.where(temp_data == self.special_atributes_from_last[i])
                temp_data = np.c_[temp_data[:, :temp_data.shape[1]-1], np.zeros(temp_data.shape[0])]
                temp_data[posiciones_a_cambiar] = 1
                np.random.shuffle(temp_data)
                temp_data = temp_data.astype('float64')

                self.data_class_list.append(data_container(temp_data, taken_percentage = 0.95)) #Añadiendo los regresores

    def rebuild_clasifier(self, in_train_set, in_test_set):
        np.random.shuffle(in_train_set)
        np.random.shuffle(in_test_set)

        self.train_set = in_train_set
        self.test_set = in_test_set #Conservamos categorías de un primer momento
        self.data_class_list.clear()

        category_1 = self.train_set[self.train_set[:, self.train_set.shape[1]-1] == self.special_atributes_from_last[0]]
        category_2 = self.train_set[self.train_set[:, self.train_set.shape[1]-1] == self.special_atributes_from_last[1]]
        category_3 = self.train_set[self.train_set[:, self.train_set.shape[1]-1] == self.special_atributes_from_last[2]]
        self.train_set_atributes = np.array([category_1, category_2, category_3])

        for i in range(0, self.special_atributes_from_last.shape[0]):
            for j in range(i+1, self.special_atributes_from_last.shape[0]):
                #El indice i representará el valor 1 mientras j será 0
                temp_data = np.concatenate((self.train_set_atributes[i], self.train_set_atributes[j]), axis = 0)
                posiciones_a_cambiar = np.where(temp_data == self.special_atributes_from_last[i])
                temp_data = np.c_[temp_data[:, :temp_data.shape[1]-1], np.zeros(temp_data.shape[0])]
                temp_data[posiciones_a_cambiar] = 1
                np.random.shuffle(temp_data)
                temp_data = temp_data.astype('float64')

                self.data_class_list.append(data_container(temp_data, taken_percentage = 0.95)) #Añadiendo los regresores

    def learning_initialization(self, iteraciones=100, alpha_learning=0.01):
        for i in range(0, len(self.data_class_list)):
            self.data_class_list[i].gradiente_function_complete_vec(iteraciones, alpha_learning, is_shown='None')

    def predict_value(self, is_shown=False):
        war_test_set = self.test_set[:, :self.test_set.shape[1]-1]
        war_test_set = war_test_set.astype('float64')
        peace_test_set = self.test_set[:, [self.test_set.shape[1]-1]]

        accerted = 0
        predicted_evaluation = np.zeros(self.special_atributes_from_last.shape[0])
        for j in range(0, war_test_set.shape[0]):
            for i in range(0, len(self.data_class_list)):
                predicted_evaluation[i] = self.data_class_list[i].sigmoidal_function(war_test_set[j])
                #-------------------------------------
                if is_shown:
                    print(i, '->', self.data_class_list[i].sigmoidal_function(war_test_set[j]))
            if is_shown:
                print(peace_test_set[j], ' -> ', np.where(self.special_atributes_from_last == peace_test_set[j]) , '\n')
            #----------------------------------------
            predicted_position_expected = int(np.where(self.special_atributes_from_last == peace_test_set[j])[0])
            who_received_more = np.zeros(self.special_atributes_from_last.shape[0])
            #who_received_more[0] = (predicted_evaluation[0] - 0.5) + (predicted_evaluation[1] - 0.5)   #MUST BE CHANGED JUST FOR THE OCASION
            #who_received_more[1] = (predicted_evaluation[2] - 0.5) + (0.5 - predicted_evaluation[0])
            #who_received_more[2] = (0.5 - predicted_evaluation[1]) + (0.5 - predicted_evaluation[2])
            if predicted_evaluation[0] < 0.5:
                who_received_more[1] +=1
            else:
                who_received_more[0] +=1

            if predicted_evaluation[1] < 0.5:
                who_received_more[2] +=1
            else:
                who_received_more[0] +=1

            if predicted_evaluation[2] < 0.5:
                who_received_more[2] +=1
            else:
                who_received_more[1] +=1

            if predicted_position_expected == np.argmax(who_received_more):
                accerted+=1
        accerted = accerted * 100 / war_test_set.shape[0]
        print('Accuracy', accerted)
        return accerted

    def specific_k_folds_formatter(self, n_folds, cat):
        if np.mod(cat.shape[0], n_folds) != 0:
            new_size = int(cat.shape[0]/ n_folds) * n_folds
            cat = cat[:new_size, :]
            return cat
        else:
            return cat

    def specific_kfolds(self, n_folds):     #Just work for this dataset
        category_1 = self.raw_data[self.raw_data[:, self.raw_data.shape[1]-1] == self.special_atributes_from_last[0]]
        category_2 = self.raw_data[self.raw_data[:, self.raw_data.shape[1]-1] == self.special_atributes_from_last[1]]
        category_3 = self.raw_data[self.raw_data[:, self.raw_data.shape[1]-1] == self.special_atributes_from_last[2]]

        category_1 = self.specific_k_folds_formatter(n_folds, category_1)
        category_2 = self.specific_k_folds_formatter(n_folds, category_2)
        category_3 = self.specific_k_folds_formatter(n_folds, category_3)

        category_1 = np.split(category_1, n_folds)
        category_2 = np.split(category_2, n_folds)
        category_3 = np.split(category_3, n_folds)

        category_1 = np.append(category_1, category_2, axis=1)
        category_1 = np.append(category_1, category_3, axis=1)
        return category_1

    def specific_kfolds_cross_validation_dev(self, train_set, test_set, iteraciones, alpha_learning):
        self.rebuild_clasifier(train_set, test_set)
        self.learning_initialization()
        return self.predict_value()

    def specific_kfolds_cross_validation(self, n_folds=3, iteraciones=500, alpha_learning=0.01):
        k_folds = self.specific_kfolds(n_folds)
        media_accuracy = 0.0
        for i in range(0, n_folds):
            temp_test_data = k_folds[i] # Data Ready
            temp_train_data = np.delete(k_folds, i, axis=0)
            temp_train_data = temp_train_data.reshape(-1, temp_test_data.shape[1]) # Data Ready
            media_accuracy += self.specific_kfolds_cross_validation_dev(temp_train_data, temp_test_data, iteraciones, alpha_learning)
        print("Accuracy promedio: ", media_accuracy/n_folds)

#************************************START HERE****************************************
data = reader_on("Iris")
data = normalize(data)

ovsa = one_vs_all(data)
ovsa.specific_kfolds_cross_validation(n_folds=3, iteraciones=3000, alpha_learning=0.4)

#ovso = one_vs_one(data)
#ovso.specific_kfolds_cross_validation(n_folds=3, iteraciones=3000, alpha_learning=0.4)
#ovso.learning_initialization()
#ovso.predict_value()
