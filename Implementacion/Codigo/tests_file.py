from my_lib import *

def main():
    np_arr = Leer_Datos("../Datos/ex1data2(Home_1f).csv")
    print(str(np_arr))
    np_arr, Media, Desviacion = Normalizar_Datos(np_arr)
    print("Datos normalizados")
    np_arr = np_arr + 1
    print(str(np_arr))
    train, test = Crear_Entrenamiento_Prueba(np_arr)
    print(train.shape)
    X_train, Y_train = separador(train)
    print(X_train.shape, Y_train.shape)
    theta = init_theta(X_train)
    print(theta)
    theta, lista_thetas, lista_costos = Gradiente_Descendiente(X_train, theta, Y_train, learning_rate = 0.4)
    #print("thetas: " + str(lista_thetas))
    #print("costos: " + str(lista_costos))
    theta2 = Ecuacion_Normal(X_train, Y_train)
    if type(theta2) != bool:
        print("Gradiente - vs - Ec. Normal")
        print(str(theta) + " - vs - " + str(theta2))
    return None

if __name__ == "__main__":
    main()
