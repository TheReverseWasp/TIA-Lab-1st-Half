from my_lib import *

def Experimento_1(fixed_datasets, sample_test):
	buffer1 = [[fixed_datasets]]
	Js = []
	for i in fixed_datasets:
		np_arr = Leer_Datos(i)
		np_arr, Media, Desviacion = Normalizar_Datos(np_arr)
		np_arr, Media, Desviacion = np_arr + 1, Media + 1, Desviacion + 1
		train, test = Crear_Entrenamiento_Prueba(np_arr, sample_test = sample_test)
		X, Y = separador(train)
		theta = Ecuacion_Normal(X, Y)
		if type(theta) != bool:
			J = Calcular_Costo(X, theta, Y)
		else:
			J = None
		Js.append(J)
	buffer1.append(Js)
	return buffer1

def Experimento_2(fixed_datasets, learning_rates):
	J_step = [500, 1000, 1500, 2000, 2500, 3000, 3500]
	statistics = {}
	for i in range(len(fixed_datasets)):
		np_arr = Leer_Datos(fixed_datasets[i])
		np_arr, Media, Desviacion = Normalizar_Datos(np_arr)
		np_arr, Media, Desviacion = np_arr + 1, Media + 1, Desviacion + 1
		train, test = Crear_Entrenamiento_Prueba(np_arr, sample_test = sample_test)
		X_train, Y_train, X_test, Y_test = separador(train), separador(test)
		for j in learning_rates:
			theta = init_theta(X_train)
			theta, lista_thetas, lista_costos_train = Gradiente_Descendiente(X_train, theta, Y_train, learning_rate = j)
			lista_costos_test = []
			for k in lista_thetas:
				lista_costos_test.append(Calcular_Costo(X_test, k, Y_test)
			statistics[str(i) + "-" + str(j) + ":train"] = lista_costos_train
			statistics[str(i) + "-" + str(j) + ":test"] = lista_costos_test
	return statistics

def Experimento_2_separados(train_dataset_name, test_dataset_name, learning_rates):
	J_step = [500, 1000, 1500, 2000, 2500, 3000, 3500]
	statistics = {}
	np_arr = Leer_Datos(train_dataset_name)
	np_test = Leer_Datos(test_dataset_name)
	np_arr, Media, Desviacion = Normalizar_Datos(np_arr)
	np_arr, np_test = np_arr + 1, Normalizar_Datos_MD(np_test, Media, Desviacion)
	Media, Desviacion = Media + 1, Desviacion + 1													#opcional
	X_train, Y_train, X_test, Y_test = separador(np_arr), separador(np_test)
	for j in learning_rates:
		theta = init_theta(X_train)
		theta, lista_thetas, lista_costos_train = Gradiente_Descendiente(X_train, theta, Y_train, learning_rate = j)
		lista_costos_test = []
		for k in lista_thetas:
			lista_costos_test.append(Calcular_Costo(X_test, k, Y_test))
		statistics["Other-" + str(j) + ":train"] = lista_costos_train
		statistics["Other-" + str(j) + ":test"] = lista_costos_test
	return statistics

def Experimento_3(sample_dataset = "../Datos/ex1data2(Home_1f).csv", learning_rate):
	np_arr = Leer_Datos(sample_dataset)
	np_arr, Media, Desviacion = Normalizar_Datos(np_arr)
	np_arr = np_arr + 1
	X_total, Y_total = separador(np_arr)
	theta_norm_general = Ecuacion_Normal(X_total, Y_total)
	if type(theta_norm_general) != bool:
		Graficar_recta_y_puntos(X_total, Y_total, theta_norm_general, "../Resultados/Exp3/Normal_total.png")
	train, test = Crear_Entrenamiento_Prueba(np_arr)
	X_train, Y_train, X_test, Y_test = separador(train), separador(test)
	theta_norm_train = Ecuacion_Normal(X_train, Y_train)
	if type(theta_norm_train) != bool:
		Graficar_recta_y_puntos(X_train, Y_train, theta_norm_train, "../Resultados/Exp3/Normal_train.png")
	theta_grad_train, __, __ = Gradiente_Descendiente(X_train, init_theta(X_train), Y_train, learning_rate = learning_rate)
	Graficar_recta_y_puntos(X_train, Y_train, theta_grad_train, "../Resultados/Exp3/Grad_train.png")

def Experimento_3_separados(train_dataset, test_dataset, learning_rate):
	