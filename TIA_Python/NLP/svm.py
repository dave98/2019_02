import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt
from sklearn import svm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from statistics import *

def loading_data(route):
    return pd.read_csv('Data/' + route + '.csv')

def k_folds_formatter(n_folds, cat):
    if np.mod(cat.shape[0], n_folds) != 0:
        new_size = int(cat.shape[0]/ n_folds) * n_folds
        cat = cat[:new_size, :]
        return cat
    else:
        return cat

def k_folds_function(n_folds, raw_data):
    category_1 = raw_data[raw_data[:, raw_data.shape[1]-1] == 1.]
    category_2 = raw_data[raw_data[:, raw_data.shape[1]-1] == 0.]

    category_1 = k_folds_formatter(n_folds, category_1)
    category_2 = k_folds_formatter(n_folds, category_2)

    category_1 = np.split(category_1, n_folds)
    category_2 = np.split(category_2, n_folds)
    category_1 = np.append(category_1, category_2, axis=1)
    return category_1

def special_k_fold_function(n_folds, raw_data):
    category_1 = raw_data[raw_data[:, raw_data.shape[1]-1] == 0.]
    category_2 = raw_data[raw_data[:, raw_data.shape[1]-1] == 1.]
    category_3 = raw_data[raw_data[:, raw_data.shape[1]-1] == 2.]


    category_1 = k_folds_formatter(n_folds, category_1)
    category_2 = k_folds_formatter(n_folds, category_2)
    category_3 = k_folds_formatter(n_folds, category_3)


    category_1 = np.split(category_1, n_folds)
    category_2 = np.split(category_2, n_folds)
    category_3 = np.split(category_3, n_folds)

    category_1 = np.append(category_1, category_2, axis=1)
    category_1 = np.append(category_1, category_3, axis=1)
    return category_1


def k_folds_cross_validation(n_folds, raw_data, is_special=False):
    np.random.shuffle(raw_data)
    if is_special:
        k_folds = special_k_fold_function(n_folds, raw_data)
    else:
        k_folds = k_folds_function(n_folds, raw_data)

    for i in range(0, n_folds):
        temp_test_data = k_folds[i]
        temp_train_data = np.delete(k_folds, i, axis=0)
        temp_train_data = temp_train_data.reshape(-1, temp_test_data.shape[1])
        np.random.shuffle(temp_train_data)

        x_train_set, y_train_set = get_x_y_data(temp_train_data)
        y_train_set = y_train_set.reshape(y_train_set.shape[0])

        x_test_set, y_test_set = get_x_y_data(temp_test_data)
        y_test_set = y_test_set.reshape(y_test_set.shape[0])

        C=3
        gamma = 1
        # Linear SVM
        linear_svm = svm.SVC(kernel = 'linear', C=C)
        linear_svm.fit(x_train_set, y_train_set)
        accuracy_linear = linear_svm.score(x_test_set, y_test_set)
        print('Accuracy Linear:', accuracy_linear)

        # Linear Polinomial
        poly_svm = svm.SVC(kernel='poly', C=C, degree=3, gamma=gamma)
        poly_svm.fit(x_train_set, y_train_set)
        accuracy_poly = poly_svm.score(x_test_set, y_test_set)
        print('Accuracy Polinomial:', accuracy_poly)

        # Linear sigmoidal
        sig_svm = svm.SVC(kernel='sigmoid', C=C, gamma=gamma)
        sig_svm.fit(x_train_set, y_train_set)
        accuracy_sig = sig_svm.score(x_test_set, y_test_set)
        print('Accuracy Sigmoidal:', accuracy_sig)

        # Linear Gaussiano
        gauss_svm = svm.SVC(kernel='rbf', C=C, gamma=gamma)
        gauss_svm.fit(x_train_set, y_train_set)
        accuracy_gauss = gauss_svm.score(x_test_set, y_test_set)
        print('Accuracy Gaussiano:', accuracy_gauss)

def get_x_y_data(in_data): # Just work for one caracteristhic
    x_data = in_data[:, :in_data.shape[1]-1]
    y_data = in_data[:, [in_data.shape[1]-1]]
    return x_data, y_data # (n, m), (n*1)[Doble array]

def normalize(in_data):
    x_data = in_data[:, :in_data.shape[1]-1]
    y_data = in_data[:, [in_data.shape[1]-1]]
    x_data = x_data.astype('float64')

    media = np.mean(x_data, axis = 0)
    sdev = np.std(x_data, axis  = 0)

    x_data = x_data - media
    x_data = np.divide(x_data, sdev)
    x_data = np.concatenate((x_data, y_data), axis = 1)
    return x_data # Raw data, preprocesed



#--------------------- For Heart or Numeric Categories ----------------------
data_container = loading_data("heart_2")
raw_data = data_container.to_numpy()
raw_data = normalize(raw_data)
k_folds_cross_validation(3, raw_data)


#--------------------- Just for Iris ----------------------
#data_container = loading_data("Iris")
#raw_data = data_container.to_numpy()
#extract_categories = raw_data[:, [raw_data.shape[1]-1]]
#extract_categories[extract_categories == 'Iris-setosa'] = 0
#extract_categories[extract_categories == 'Iris-versicolor'] = 1
#extract_categories[extract_categories == 'Iris-virginica'] = 2
#extract_categories = extract_categories.astype('float64')

#raw_data = raw_data[:, :raw_data.shape[1]-1]
#raw_data = np.c_[raw_data, extract_categories]
#raw_data = raw_data.astype('float64')
#raw_data = normalize(raw_data)

#k_folds_cross_validation(3, raw_data, is_special=True)


#'Iris-setosa'
#'Iris-versicolor'
#'Iris-virginica'
