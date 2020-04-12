import numpy as np
import pandas as pd
from sklearn import preprocessing

####Implementacion
#1
def Leer_Datos(filename):
    content = pd.read_csv(filename, encoding = 'ascii', dtype = np.float64)
    #X, Y = content.to_numpy().T[:-1], content.to_numpy().T[-1]
    #return X, Y
    return content

#complementaria
def Normalizar_Datos_np(npmatrix):
    npmins1 = np.asarray([[np.amin(row) for row in npmatrix]]).T                 #Hallo minimos
    normst1 = npmatrix - npmins1                                                 #Paso 1: resta vectorizada de minimos
    npmaxs1 = np.asarray([[np.amax(row) for row in normst1]]).T                  #Hallo maximos
    npmaxs1 /= 2                                                                 #Aseguro rango [-1,1]
    normst2 = normst1 / npmaxs1                                                 #Paso 2: divido sobre maximos
    return normst2, npmins1, npmaxs1

#2 pd
def Normalizar_Datos_pd(content):
    min_max_scaler = preprocessing.MinMaxScaler()
    content_scaled = min_max_scaler.fit_transform(content)
    df = pd.DataFrame(content_scaled)
    return df

#3
def Crear_Entrenamiento_Prueba(df, frac = 0.7):
    msk = np.random.randn(len(df)) <= frac
    train = df[msk]
    test = df[-msk]
    return train, test

#complementaria
def Distribucion_a_Numpy(df):
    X, Y = df.to_numpy().T[:-1], df.to_numpy().T[-1]
    return X, Y

#4
def Calcular_Costo(X, Y, theta):
    m = X.shape[1]
    J = np.power(np.sum(X * theta[0,:-1].T + theta[0,-1], axis = 1, keepdims = True) - Y, 2) / (2 * m)
    return J

#complementaria
def Inicializar_thetas(X):
    theta = np.random.randn(X.shape[0] + 1)
    theta[-1] = 0
    return theta

#5
def Gradiente_Descendiente(X, Y, theta, learning_rate, num_iterations):
    m = X.shape[0]
    lista_costos = []
    for i in range(num_iterations):
        theta[0,:-1] -= learning_rate * (np.sum(X * theta[0,:-1].T + theta[0,-1], axis = 1, keepdims = True) - Y + X) / m
        theta[0,-1] -= learning_rate * (np.sum(X * theta[0,:-1].T + theta[0,-1], axis = 1, keepdims = True) - Y) / m
        J = Calcular_Costo(X, Y, theta)
        if i % 100 == 0:
            print("Costo en la Iteración " + str(i) + ": " + str(J))
            lista_costos.append(J)
    return theta, lista_costos

#6
def Ecuacion_Normal(X, Y, learning_rate = 0.5, num_iterations = 1001):
    #por definir
    return True

####Experimentos
#Prediccion
class Predictor:
    def __init__(self):
        self.dataset_path = ["../Datos/ex1data2(Home_1f).csv" , "../Datos/oceano_simple.csv", "../Datos/petrol_consumption.csv"]
        self.dataset_name = ["Precio, Viviendas", "Temperatura Oceano", "Consumo de Petroleo"]

    def set_data_train_test(self):
        self.train0, self.test0 = Crear_Entrenamiento_Prueba(Normalizar_Datos_pd(Leer_Datos(self.dataset_path[0])))
        self.train1, self.test1 = Crear_Entrenamiento_Prueba(Normalizar_Datos_pd(Leer_Datos(self.dataset_path[1])))
        self.train2, self.test2 = Crear_Entrenamiento_Prueba(Normalizar_Datos_pd(Leer_Datos(self.dataset_path[2])))

    def Experimento_1(self):

    def Experimento_2(self):

    def Experimento_3(self):                    #grafica datos train test Precio Vivienda

    def Experimento_4(self):                    #graficas de costo 3 datasets


def Experimento_1():
