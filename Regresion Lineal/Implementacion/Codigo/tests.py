import numpy as np
import pandas as pd
from lib_reglineal import *

def main():
    #print(1)
    #print(str(Leer_Datos("../Datos/ex1data2(Home_1f).csv")))
    print(2)
    #print(str(Normalizar_Datos_pd(Leer_Datos("../Datos/oceano_simple.csv"))))
    #print(3)
    #print(str(Leer_Datos("../Datos/petrol_consumption.csv")))

    #Oceano
    #data = Normalizar_Datos_pd(Leer_Datos("../Datos/oceano_simple.csv"))
    #Consumo de petroleo
    data = Normalizar_Datos_pd(Leer_Datos("../Datos/petrol_consumption.csv"))
    #Precio vivienda
    #data = Normalizar_Datos_pd(Leer_Datos("../Datos/ex1data2(Home_1f).csv"))
    train, test = Crear_Entrenamiento_Prueba(data)
    X_train, Y_train = Distribucion_a_Numpy(train)
    #print(str(X_train), str(Y_train))
    X_test, Y_test = Distribucion_a_Numpy(test)
    theta = Inicializar_thetas(X_train)
    print(theta)
    theta, lista_costos = Gradiente_Descendiente(X_train, Y_train, theta)
    Y_hat, mean_error = Predecir(X_test, Y_test, theta)
    print("mean error: " + str(mean_error))
    print(theta)
    print(str(Y_test) + "\n" + str(Y_hat))



if __name__ == "__main__":
    main()
