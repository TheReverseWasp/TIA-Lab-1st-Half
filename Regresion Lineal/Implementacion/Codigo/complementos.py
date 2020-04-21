import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

epsilon = 1e-10

def media(np_arr):
    Media = np.sum(np_arr, keepdims = True, axis = 1)
    Media = Media / np_arr.shape[1]
    return Media

def desviacion_estandar_1(np_arr):
    Media = media(np_arr)
    Desviacion = (np.sum(np_arr - Media, keepdims = True, axis = 1) / np_arr.shape[1]) ** 0.5 + epsilon
    return Media, Desviacion

def desviacion_estandar_2(np_arr):
    Media = media(np_arr)
    Desviacion = np.max(np_arr, keepdims = True, axis = 1) - np.min(np_arr, keepdims = True, axis = 1) + epsilon
    return Media, Desviacion

def separador(np_arr):
    X, Y = np_arr[:-1], np.asarray([np_arr[-1]])
    return X, Y

def init_theta(X):
    theta = np.zeros((1, X.shape[0]))
    return theta

def Prediccion_Regresion_Lineal(X, theta):
    Y_hat = np.dot(theta, X)
    return Y_hat

def check_type(_type):
    opcion = input()
    pass_code = False
    while not pass_code:
        try:
            opcion = _type(opcion2)
            pass_code = True
        except:
            print("Error: Entrada Erronea, ingrese la entrada nuevamente")
            opcion = input()
    return opcion

def Graficar_recta_y_puntos(X, Y, theta, path_to_save):
    X_line = np.arange(0., 2.7, .1)
    Y_line = theta[0][0] + theta[0][1] * X_line
    plt.plot(X_line, Y_line)
    plt.scatter(X[1],Y[0])
    plt.savefig(path_to_save)

def Graficar_Costo(lista_Costos, path_to_save):
    X_line = np.arange(0, 3501, 500)
    Y_line = lista_Costos
    plt.plot(X_line, Y_line)
    plt.savefig(path_to_save)