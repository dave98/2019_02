import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#-------------------------------FUNCIONES SECUNDARIAS--------------------------------
def extended_print(in_array):
    ncolumns = in_array[0].size
    nfilas = int(in_array.size/ncolumns)
    for i in range(0, nfilas):
        print(in_array[i], '\n')


#-------------------------------FUNCIONES PRIMARIAS-----------------------------------
def reader_on(route):
    data = pd.read_csv("Data/" + route + ".csv")
    data = data.to_numpy()
    return data
    #print("Data/" + route + ".csv")

def normalize(in_data): # Todo menos la última fila, movimiento aleatorio
    x_data = in_data[:, :in_data.shape[1]-1] #Filas, columnas
    y_data = in_data[:,[in_data.shape[1]-1]]
    x_data = x_data.astype('float64') #Asugrando tipo de dato

    media = np.mean(x_data, axis = 0)
    sdev = np.std(x_data, axis = 0)

    x_data = x_data - media
    x_data = np.divide(x_data, sdev)
    x_data = np.concatenate((x_data, y_data), axis = 1)
    x_data = np.c_[np.ones(x_data.shape[0]), x_data] # Añadiendo columna de unos al inicio, tipo de dato float64 correcto
    return x_data

class data_container:
    def __init__(self, in_data, taken_percentage = 0.85):
        taken_rows = int(in_data.shape[0] * taken_percentage)
        self.raw_data = in_data.copy() # Guardamos los datos para operaciones posteriores

        np.random.shuffle(in_data)
        self.x_train = in_data[:taken_rows-1, :in_data.shape[1]-1]
        self.y_train = in_data[:taken_rows-1, [in_data.shape[1]-1]]
        self.y_train = self.y_train.reshape(self.y_train.size)

        self.x_test = in_data[taken_rows-1:in_data.shape[0], :in_data.shape[1]-1]
        self.y_test = in_data[taken_rows-1:in_data.shape[0], [in_data.shape[1]-1]]
        self.y_test = self.y_test.reshape(self.y_test.size)

        self.thetha_array = np.zeros((self.x_train.shape[1]), dtype=np.float64)
        self.thetha_array = np.random.rand(self.x_train.shape[1])
        self.alpha_learning = 0.01 # DEFAULT
        self.n_iteracciones = 100
        self.cost_history = np.zeros(self.n_iteracciones, dtype=np.float64)
        self.thetha_history = np.zeros((self.n_iteracciones, self.x_train.shape[1])) #Numero de iteracciones por cantidad de thethas
        self.predicted_history = np.zeros(self.x_test.shape[1], dtype=np.int16)

    def reset_data_container(self, in_train, in_test):
        np.random.shuffle(in_train)
        np.random.shuffle(in_test)

        self.x_train = in_train[:, :in_train.shape[1]-1]
        self.y_train = in_train[:, [in_train.shape[1]-1]]
        self.y_train = self.y_train.reshape(self.y_train.size)

        self.x_test = in_test[:, :in_test.shape[1]-1]
        self.y_test = in_test[:, [in_test.shape[1]-1]]
        self.y_test = self.y_test.reshape(self.y_test.size)

        self.thetha_array = np.zeros((self.x_train.shape[1]), dtype=np.float64)
        #self.thetha_array = np.random.rand(self.x_train.shape[1])

    def set_train_parameters(self, in_iteracciones, in_alpha_learning):
        self.alpha_learning = in_alpha_learning
        self.n_iteracciones = in_iteracciones
        self.cost_history = np.zeros(self.n_iteracciones, dtype=np.float64)
        self.thetha_history = np.zeros( (self.n_iteracciones, self.x_train.shape[1]), dtype=np.float64)
        self.predicted_history = np.zeros(self.x_test.shape[1], dtype=np.int16)
        self.thetha_array = np.zeros((self.x_train.shape[1]), dtype=np.float64)

    def describe(self):
        print('Iteraciones propuestas: ', self.n_iteracciones)
        print('Margen de aprendizaje: ', self.alpha_learning)
        print('x Train: \n', self.x_train)
        print('y Train: \n', self.y_train)
        print('x Test: \n', self.x_test)
        print('y Test: \n', self.y_test)
        print('Thetha array: \n', self.thetha_array)

#++++++++++++++++++++++++++++++++++++++++++++TRADITIONAL IMPLEMENTATION+++++++++++++++++++++++++++++++++++++++
    def h_function(self, x_singular_array): #Usamos X0 para igualar tamaños de array en theta debido a tetha zero. x0 SIEMPRE va a ser uno
        r_array = np.multiply(self.thetha_array, x_singular_array)
        result = np.sum(r_array)
        return result

    def sigmoidal_function(self, x_singular_array): #Recibe array de thetas y X de un conjunto de prueba o entrenamiento y retorna su aplicación bajo la función sigmoidal
        h_value = self.h_function(x_singular_array)
        result = 1 / (1 + np.exp(-1 * h_value))
        return result

    def cost_function(self, x_singular_array, y_singular_array): # y_singular_array es solo un numero
        f_cost = 0.0
        m = self.x_train.shape[0]
        for i in range(0, m):
                f_cost = f_cost +  (    (self.y_train[i] * np.log2(self.sigmoidal_function(self.x_train[i]))) + ((1-self.y_train[i]) * np.log2(1-sigmoidal_function(self.x_train[i])))   )
        f_cost = (-1/m) * f_cost
        return f_cost

    def gradiente_function_dev(self, j_thetha_used):
        f_result = 0.0
        m = self.x_train.shapex[0]
        for i in range(0, m):
            f_result = f_result + ((self.sigmoidal_function(self.x_train[i]) - self.y_train[i]) * self.x_train[i, j_thetha_used] )
        f_result = self.alpha_learning * f_result
        return f_result

    def gradiente_function_singular(self): #Una iteracción
        alternative_thetha = self.thetha_array
        for i in range(0, self.thetha_array.shape[0]):
            alternative_thetha[i] = alternative_thetha[i] - self.gradiente_function_dev(i)
        self.thetha_array = alternative_thetha

    def gradiente_function_complete(self, n_iteracciones, alpha_learning, is_shown=False):
        print('Entrenando bajo:', n_iteracciones, 'iteracciones con tasa de arendizaje de', alpha_learning)
        print('Original tetha', self.thetha_array, '\n')
        self.set_train_parameters(n_iteracciones, alpha_learning) # Configuramos nuevos parámetros de entramiento
        for i in range(0, n_iteracciones):
            self.gradiente_function_singular()
            if is_shown:
                print('Iteraccion ', i, 'con alpha', self.thetha_array)
        print('Final Thetha:', self.thetha_array)

#//////////////////////////////////////////VECTORIAL TREAMENT////////////////////////////////////////////
    def h_function_vec(self): #Trabaja para todos los x de entrenamiento <- TESTED
        return np.multiply(self.thetha_array, self.x_train) #retorna un matrix de n(vertical) n->numero de muestras de entrenamiento x m -> numero de caracteristicas por x

    def sigmoidal_function_vec(self): #TESTED
        result = np.sum(self.h_function_vec(), axis=1) # Recibe todo el conjunto de entrenamiento x multiplicado por thetha, suma axis X
        return 1 / (1 + np.exp(-1 * result)) # Retorna todos los x pasados por la sigmoidal en una matriz de 1 x m donde m es la cantidad de datos de entrenamiento

    def sigmoidal_function_for_test_vec(self): #Evalúa el conjunto de prueba con la función sigmoidal
        result = np.multiply(self.thetha_array, self.x_test)
        result = np.sum(result, axis=1)
        return 1 / (1 + np.exp(-1 * result))

    def cost_function_vec(self): #Taken from page 54, slides. Retorna un valor con el costo actual de la aplicación thetha, datos de entrenamiento
        sigmoidal_temp = self.sigmoidal_function_vec()
        result = np.multiply(np.log(sigmoidal_temp), self.y_train) + ((1 - self.y_train)*(np.log(1 - sigmoidal_temp)))
        result = np.sum(result)
        return -1/self.x_train.shape[0] * result

    def gradiente_function_singular_vec(self):
        alternative_thetha = self.thetha_array
        m = self.x_train.shape[0]
        #f_result = self.alpha_learning * np.sum(self.sigmoidal_function_vec() - self.y_train)
        f_result = (self.sigmoidal_function_vec() - self.y_train) # Nx1 --- NxM -> Nxm
        f_result = f_result.reshape(f_result.shape[0], 1)
        f_result = f_result * self.x_train
        f_result = (self.alpha_learning/m) * np.sum(f_result, axis=0)

        self.thetha_array = alternative_thetha - f_result

    def gradiente_function_complete_vec(self, n_iteracciones, alpha_learning, is_shown='None'):
        if is_shown == 'Minimal':
            print('Entrenando bajo:', n_iteracciones, 'iteracciones con tasa de arendizaje de', alpha_learning)
            print('Original tetha', self.thetha_array)
        self.set_train_parameters(n_iteracciones, alpha_learning) # Configuramos nuevos parámetros de entramiento
        for i in range(0, n_iteracciones):
            self.gradiente_function_singular_vec()
            self.cost_history[i] = self.cost_function_vec()
            self.thetha_history[i] = self.thetha_array
            if is_shown == 'Complete':
                print('Iteraccion ', i, 'con thetha as:', self.thetha_array)
        if is_shown == 'Minimal':
            print('Final Thetha:', self.thetha_array)
            print('Cost Evolution: ', self.cost_history)

    def clasifier_function(self): #Trabajo con el conjunto de entrenamiento guardado en la clase
        total_data_analyzed = self.x_test.shape[0]
        for i in range(0, self.x_test.shape[0]):
            temp_result = self.sigmoidal_function(self.x_test[i])
            if temp_result >= 0.5:
                temp_result = 1.0
            else:
                temp_result = 0.0
            print(temp_result, '<->', self.y_test[i])

    def clasifier_function_vec(self):
        predicted_data = self.sigmoidal_function_for_test_vec()
        predicted_data = predicted_data.round()
        self.predicted_data = predicted_data

    def accuracy_analyzer_vec(self):
        number_of_equal_elements = np.sum(self.predicted_data == self.y_test)
        total_elements = self.y_test.shape[0]

        percentaje = number_of_equal_elements * 100 / total_elements
        print('Porcentaje de equivalencia: ', percentaje)
        return percentaje

    #------------------------------Exercises Methods------------------------------------
    def k_folds_formatter(self, n_folds, cat):
        if np.mod(cat.shape[0], n_folds) != 0:
            new_size = int(cat.shape[0]/ n_folds) * n_folds
            cat = cat[:new_size, :]
            return cat
        else:
            return cat

    def k_folds_function(self, n_folds): #Separa los datos en kfolds
        category_1 = self.raw_data[self.raw_data[:, self.raw_data.shape[1]-1] == 1.]
        category_2 = self.raw_data[self.raw_data[:, self.raw_data.shape[1]-1] == 0.]

        category_1 = self.k_folds_formatter(n_folds, category_1)
        category_2 = self.k_folds_formatter(n_folds, category_2)

        category_1 = np.split(category_1, n_folds)
        category_2 = np.split(category_2, n_folds)
        category_1 = np.append(category_1, category_2, axis=1)
        return category_1

    def k_folds_cross_validation_dev(self, train_set, test_set, iteraciones, alpha_learning):
        self.reset_data_container(train_set, test_set)
        self.gradiente_function_complete_vec(iteraciones, alpha_learning)
        self.clasifier_function_vec()
        return self.accuracy_analyzer_vec()

    def k_folds_cross_validation(self, n_folds=3, iteraciones=100, alpha_learning=0.01): #Iterracciones de kfolds
        k_folds = self.k_folds_function(n_folds)
        media_accuracy = 0
        for i in range(0, n_folds):
            temp_test_data = k_folds[i] # Data Ready
            temp_train_data = np.delete(k_folds, i, axis=0)
            temp_train_data = temp_train_data.reshape(-1, temp_test_data.shape[1]) # Data Ready
            media_accuracy += self.k_folds_cross_validation_dev(temp_train_data, temp_test_data, iteraciones, alpha_learning)
        print("Accuracy promedio: ", media_accuracy/n_folds)

#--------------------------------------Experiment-----------------------------
    def experiment_1(self):
        for i in range(0, 6):
            ritmo_aprendizaje = float(input('Introducir tasa de aprendizaje: '))
            total_iteraciones = int(input('Numero de iteraciones: '))
            self.k_folds_cross_validation(n_folds=3, iteraciones=total_iteraciones, alpha_learning=ritmo_aprendizaje)

    def experiment_2(self, iteraciones, alpha_learning, title):
        self.gradiente_function_complete_vec(iteraciones, alpha_learning)
        d_admin.clasifier_function_vec()
        d_admin.accuracy_analyzer_vec()

        plot_array = d_admin.cost_history
        plt.plot(plot_array)
        plt.suptitle(title)
        plt.ylabel('Costo')
        plt.xlabel('Iteracción')
        plt.show()

#---------------------------------START HERE-----------------------------------------
#data = reader_on("diabetes_2") #Leendo archivo
#data = normalize(data)

#d_admin = data_container(data)
#d_admin.gradiente_function_complete_vec(200, 0.1, is_shown=False)
#d_admin.clasifier_function_vec()
#d_admin.accuracy_analyzer_vec()
#d_admin.k_folds_cross_validation(n_folds=3, iteraciones=1000)
#d_admin.experiment_1()
#d_admin.experiment_2(iteraciones=500, alpha_learning=0.1, title='Diabetes')

















#########################
