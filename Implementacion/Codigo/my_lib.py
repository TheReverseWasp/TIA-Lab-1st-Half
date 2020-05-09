from complementos import *

#1
def Leer_Datos(filename):
    df = pd.read_csv(filename, sep = "\t")
    np_arr = df.to_numpy()
    np_arr = np_arr.T
    temp = [np.ones(np_arr.shape[1]).tolist()]
    for i in np_arr:
        temp.append(i.tolist())
    answer = np.asarray(temp)
    return answer

#2
def Normalizar_Datos(np_arr):
    Media, Desviacion = desviacion_estandar_2(np_arr)
    answer = (np_arr - Media) / Desviacion
    return answer, Media, Desviacion

def Normalizar_Datos_MD(np_arr, Media, Desviacion):
    answer = (np_arr - Media) / Desviacion
    return answer

#3
def Crear_Entrenamiento_Prueba(np_arr, sample_test = 0.7):
    df = pd.DataFrame(data = np_arr.T)
    msk = np.random.randn(len(df)) <= sample_test
    train = df[msk]
    test = df[~msk]
    train = train.to_numpy().T
    test = test.to_numpy().T
    return train, test

#4
def Calcular_Costo(X, theta, Y):
    Costo = (np.sum(np.dot(theta, X) - Y) **2) / (2 * X.shape[1])
    return Costo

#5
def Gradiente_Descendiente(X, theta, Y, learning_rate =0.4, iteraciones = 3501, interval = 500):
    lista_thetas = []
    lista_costos = []
    m = X.shape[1]
    for i in range(iteraciones):
        theta = theta - learning_rate * np.sum((np.dot(theta, X) - Y)*X, keepdims = True, axis = 1).T / m
        if i % interval == 0:
            J = Calcular_Costo(X, theta, Y)
            #print("Costo en la iteracion " + str(i) + ": " + str(J))
            #print("thetas en la iteracion " + str(i) + ": " + str(theta))
            lista_thetas.append(theta)
            lista_costos.append(J)
    return theta, lista_thetas, lista_costos

#6
def Ecuacion_Normal(X, Y):
    try:
        theta = np.dot(np.dot(np.linalg.inv(np.dot(X, X.T)), X), Y.T).T
        return theta
    except:
        print("Error: Matriz Singular")
        return False
