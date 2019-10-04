import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#NOTA: Cantidad de thetas igual al numero de columnas en train_x + 1
def apply_function(train_x_singular_entry, thetha_array):
    #Theta array debe contener un elemento más que train_x_singular_entry
    #NOTA: Existe conservación de tipo a pesar de instanciación normal
    theta_zero = thetha_array[0]
    r_array = thetha_array[1:]
    r_array = np.multiply(train_x_singular_entry, r_array)
    result = np.sum(r_array) + theta_zero
    return result

def cost_function(train_x, train_y, test_x, test_y, thetha_array):
    f_cost = 0.0
    m = int(train_x.size/train_x[0].size) #Numero de muestras en conjunto de entrenamiento. Igual en X que en Y
    for i in range(0, m):
        f_cost = f_cost + np.power(( apply_function(train_x[i], thetha_array) - train_y[i]), 2)
    f_cost = (1/(2*m))*f_cost
    return f_cost

#-------------------------[    DATA       ] [lista de thetas] [Selected feature]     [Ritmo]
def multiple_cost_function(train_x, train_y,   thetha_array,    theta_designed,   learning_rate, is_theta_zero=False):
    f_cost = 0.0
    m = int(train_x.size/train_x[0].size)
    if is_theta_zero:
        for i in range(0, m):
            f_cost = f_cost + (apply_function(train_x[i], thetha_array) - train_y[i])
    else:
        for i in range(0, m):
            f_cost = f_cost + ((apply_function(train_x[i], thetha_array) - train_y[i]) * train_x[i, theta_designed]) #Train_x esta bien señalado
    f_cost = (learning_rate/m) * f_cost
    return f_cost

def theta_process(train_x, train_y, learning_rate, thetha_array):
    #sum_cost = multiple_cost_function(train_x, train_y, thetha_array, 0, learning_rate, True) #LINEA IRRELEVANTE, utilizada para cálculo de costo
    #thetha_array[0] = thetha_array[0] - multiple_cost_function(train_x, train_y, thetha_array, 0, learning_rate, True) # No importa theta_designed, no se utiliza con True
    temp_thetha_array = np.zeros(thetha_array.size, dtype=np.float64)
    temp_thetha_array[0] = thetha_array[0] - multiple_cost_function(train_x, train_y, thetha_array, 0, learning_rate, True)
    for i in range(1, thetha_array.size):
        #thetha_array[i] = thetha_array[i] - multiple_cost_function(train_x, train_y, thetha_array, i-1, learning_rate) # i-1 Puesto que hay una caracteristica menos que el numero de thetas y no se usa el theta zero
        #sum_cost = sum_cost + multiple_cost_function(train_x, train_y, thetha_array, i-1, learning_rate) #LINEA IRRELEVANTE, utilizada para cálculo de costo
        temp_thetha_array[i] = thetha_array[i] - multiple_cost_function(train_x, train_y, thetha_array, i-1, learning_rate)
    thetha_array = np.copy(temp_thetha_array)
    return thetha_array

def grad_descendiente(train_x, train_y, n_iteracciones, learning_rate):
    thetha_array = np.zeros( (train_x[0].size + 1) , dtype=np.float64) # Creando array theta inicial con caracteristicas + 1 thethas. Inicia en zero
    for learn_it in range(0, n_iteracciones):
        thetha_array = theta_process(train_x, train_y, learning_rate, thetha_array)
    return thetha_array

#Recibe los valores thethas ya definidos y un ejemplar de prueba para posteriormente devolver el resultado
def multiple_test_evaluation(thetha_array, test_x_singular_array):
    #print(thetha_array, '\n', test_x_singular_array)
    m = test_x_singular_array.size
    f_result = thetha_array[0]
    for i in range(0, m):
        f_result = f_result + (thetha_array[i+1] * test_x_singular_array[i])
    return f_result

#Resumen del experimento 1 solicitado en la tarea: Envia conjunto de prueba x y desarrolla la funcion con los thethas encontrados.
#Posteriormente se evalúa los resultados obtenidos con el conjunto de prube y y se obtiene el error cuadratico medio
def experiment_1(thetha_array, test_x_entire_set, test_y_entire_set):
    n_test_size = int(test_x_entire_set.size/test_x_entire_set[0].size)
    test_results_pool = np.zeros( (n_test_size) , dtype=np.float64)
    for i in range(0, n_test_size):
        #print(multiple_test_evaluation(thetha_array, test_x_entire_set[i]), ' <-> ', test_y_entire_set[i])
        test_results_pool[i] = multiple_test_evaluation(thetha_array, test_x_entire_set[i])

    mean_square_error = (np.power( test_y_entire_set - test_results_pool, 2)).mean()
    print('Mean squared error: ', mean_square_error)
    #Printing Service
    for i in range(0, n_test_size):
        print(test_x_entire_set[i], test_y_entire_set[i], test_results_pool[i])

def experiment_2(thetha_array, test_x_entire_set, test_y_entire_set):
    n_test_size = int(test_x_entire_set.size/test_x_entire_set[0].size)
    test_results_pool = np.zeros( (n_test_size) , dtype=np.float64)
    for i in range(0, n_test_size):
        test_results_pool[i] = multiple_test_evaluation(thetha_array, test_x_entire_set[i])

    mean_square_error = (np.power( test_y_entire_set - test_results_pool, 2)).mean()
    print('Mean squared error: ', mean_square_error)

def experiment_4(thetha_array, train_x_entire_set, train_y_entire_set, test_x_entire_set, test_y_entire_set):
    #Iniciamos con los datos de entrenamiento, thetha array mínimo de dos

    train_m = int(train_x_entire_set.size/train_x_entire_set[0].size) # Elementos del conjunto de entrenamiento
    thetha_iteraction = 50 #Izquiera derecha, total cien perturbaciones a la recta
    thetha_perturbation = 0.05

    # Los thethas inician con una perturbación de thetha_iteraction*thetha_perturbation. Dicha perturbación ira desapareciendo hasta cero para después crecer
    thetha_remanent = thetha_array[0] #Thetha solitario permanece inalterable, usado a lo largo de toda la ejecución
    thetha_array = thetha_array - (thetha_perturbation * thetha_iteraction)
    thetha_array[0] = thetha_remanent

    #Grupo de entrenamiento
    trainxplot = np.zeros( int(thetha_iteraction*2))
    trainyplot = np.zeros( int(thetha_iteraction*2))

    for j in range(0, 2*thetha_iteraction):
        n_temp = 0.0 #Temporal no afectable
        for i in range(0, train_m):
            n_temp = n_temp + np.power((apply_function(train_x_entire_set[i], thetha_array) - train_y_entire_set[i]), 2)
        n_temp = (1/(2*train_m)) * n_temp
        trainxplot[j] = thetha_array[1] # Almacenamos los valores divergentes
        trainyplot[j] = n_temp
        # Añadiendo perturbación en thethas
        thetha_array = thetha_array + thetha_perturbation
        thetha_array[0] = thetha_remanent

    #Grupo de Prueba
    test_m = int(test_x_entire_set.size/test_x_entire_set[0].size) # Elemento del conjunto de prueba

    thetha_array = thetha_array - (2*thetha_perturbation * thetha_iteraction)
    thetha_array[0] = thetha_remanent

    testxplot = np.zeros(int(thetha_iteraction*2))
    testyplot = np.zeros(int(thetha_iteraction*2))

    for j in range(0, 2*thetha_iteraction):
        n_temp = 0.0
        for i in range(0, test_m):
            n_temp = n_temp + np.power((apply_function(test_x_entire_set[i], thetha_array) - test_y_entire_set[i]), 2)
        n_temp = (1/(2*test_m)) * n_temp
        testxplot[j] = thetha_array[1]
        testyplot[j] = n_temp
        # Añadiendo perturbación en thethas
        thetha_array = thetha_array + thetha_perturbation
        thetha_array[0] = thetha_remanent


    plt.plot(trainxplot, trainyplot, label="Datos de Entrenamiento")
    plt.plot(testxplot, testyplot, label="Datos de Prueba")
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.title('Función de Costo')
    plt.axis([-3.5, 1.5, 0, 3])
    plt.legend()
    plt.show()
