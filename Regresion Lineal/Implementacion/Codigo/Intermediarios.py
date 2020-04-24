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
	print(theta)
	return buffer1

def Experimento_1_X(global_name, train_name, test_name):
	Js = []
	np_arr = Leer_Datos(global_name)
	np_arr, Media, Desviacion = Normalizar_Datos(np_arr)
	train_data = Leer_Datos(train_name)
	test_data = Leer_Datos(test_name)
	train, test = Normalizar_Datos_MD(train_data, Media, Desviacion),  Normalizar_Datos_MD(test_data, Media, Desviacion)
	train, test = train + 1, test + 1
	X_train, Y_train = separador(train)
	X_test, Y_test = separador(test)
	theta_train = Ecuacion_Normal(X_train, Y_train)
	theta_test = Ecuacion_Normal(X_test, Y_test)
	if type(theta_train) != bool or type(theta_test):
		J = Calcular_Costo(X_train, theta_train, Y_train)
		Js.append(J)
		J = Calcular_Costo(X_test, theta_test, Y_test)
		Js.append(J)
	else:
		J = None
	buffer1 = []
	buffer1.append(Js)
	print("train: " + str(theta_train))
	print("test: " + str(theta_test))
	return buffer1

def Experimento_2(fixed_datasets, learning_rates):
	J_step = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500]
	print("step: " + str(J_step))
	statistics = {}
	for i in range(len(fixed_datasets)):
		np_arr = Leer_Datos(fixed_datasets[i])
		np_arr, Media, Desviacion = Normalizar_Datos(np_arr)
		np_arr, Media, Desviacion = np_arr + 1, Media + 1, Desviacion + 1
		train, test = Crear_Entrenamiento_Prueba(np_arr)
		(X_train, Y_train), (X_test, Y_test) = separador(train), separador(test)
		for j in learning_rates:
			theta = init_theta(X_train)
			theta, lista_thetas, lista_costos_train = Gradiente_Descendiente(X_train, theta, Y_train, learning_rate = j)
			lista_costos_test = []
			for k in lista_thetas:
				lista_costos_test.append(Calcular_Costo(X_test, k, Y_test))
			statistics[str(i) + "-" + str(j) + ":train"] = lista_costos_train
			statistics[str(i) + "-" + str(j) + ":test"] = lista_costos_test
	return statistics

def Experimento_2_separados(train_dataset_name, test_dataset_name, learning_rates):
	J_step = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500]
	print("step: " + str(J_step))
	statistics = {}
	try:
		np_arr = Leer_Datos(train_dataset_name)
		np_test = Leer_Datos(test_dataset_name)
	except:
		print("Error: Los nombres de los datasets son incorrectos")
		return {}
	np_arr, Media, Desviacion = Normalizar_Datos(np_arr)
	np_arr, np_test = np_arr + 1, Normalizar_Datos_MD(np_test, Media, Desviacion)
	Media = Media + 1																				#opcional
	(X_train, Y_train), (X_test, Y_test) = separador(np_arr), separador(np_test)
	for j in learning_rates:
		theta = init_theta(X_train)
		theta, lista_thetas, lista_costos_train = Gradiente_Descendiente(X_train, theta, Y_train, learning_rate = j)
		lista_costos_test = []
		for k in lista_thetas:
			lista_costos_test.append(Calcular_Costo(X_test, k, Y_test))
		statistics["Other-" + str(j) + ":train"] = lista_costos_train
		statistics["Other-" + str(j) + ":test"] = lista_costos_test
	return statistics

def Experimento_3(learning_rate, iteraciones, sample_dataset = "../Datos/ex1data2(Home_1f).csv"):
	np_arr = Leer_Datos(sample_dataset)
	np_arr, Media, Desviacion = Normalizar_Datos(np_arr)
	np_arr = np_arr + 1
	X_total, Y_total = separador(np_arr)
	theta_norm_general = Ecuacion_Normal(X_total, Y_total)
	if type(theta_norm_general) != bool:
		Graficar_recta_y_puntos(X_total, Y_total, theta_norm_general, "../Resultados/Exp3/Normal_total.png")
	train, test = Crear_Entrenamiento_Prueba(np_arr)
	(X_train, Y_train), (X_test, Y_test) = separador(train), separador(test)
	theta_norm_train = Ecuacion_Normal(X_train, Y_train)
	if type(theta_norm_train) != bool:
		Graficar_recta_y_puntos(X_train, Y_train, theta_norm_train, "../Resultados/Exp3/Normal_train.png")
	theta_grad_train, __, __ = Gradiente_Descendiente(X_train, init_theta(X_train), Y_train, learning_rate = learning_rate, iteraciones = iteraciones)
	Graficar_recta_y_puntos(X_train, Y_train, theta_grad_train, "../Resultados/Exp3/Grad_train.png")

def Experimento_3_separados(train_dataset, test_dataset, learning_rate, iteraciones):
	try:
		train = Leer_Datos(train_dataset)
		test = Leer_Datos(test_dataset)
	except:
		print("Error: Los nombres de los datasets son incorrectos")
		return
	train, Media, Desviacion = Normalizar_Datosa(train)
	train = train + 1
	test = Normalizar_Datos_MD(test, Media, Desviacion)
	test = test + 1
	(X_train, Y_train), (X_test, Y_test) = separador(train), separador(test)
	theta_norm = Ecuacion_Normal(X_train, Y_train)
	Graficar_recta_y_puntos(X_train, Y_train, theta_norm, "../Resultados/Exp3/Norm_other_train.png")
	theta_grad, __, __ = Gradiente_Descendiente(X_train, init_theta(X_train), Y_train, learning_rate = learning_rate, iteraciones = iteraciones)
	Graficar_recta_y_puntos(X_train, Y_train, theta_grad, "../Resultados/Exp3/Grad_other_train.png")
	Graficar_recta_y_puntos(X_test, Y_test, theta_grad, "../Resultados/Exp3/Grad_other_test.png")
	
def Experimento_4(fixed_datasets, learning_rate):
	for i in range(len(fixed_datasets)):
		np_arr = Leer_Datos(fixed_datasets[i])
		np_arr, __, __ = Normalizar_Datos(np_arr)
		np_arr = np_arr + 1
		train, test = Crear_Entrenamiento_Prueba(np_arr)
		(X_train, Y_train), (X_test, Y_test) = separador(train), separador(test)
		theta = init_theta(X_train)
		theta, lista_thetas, lista_costos_train = Gradiente_Descendiente(X_train, theta, Y_train, learning_rate = learning_rate, interval = 50)
		lista_costos_test = []
		for j in lista_thetas:
			lista_costos_test.append(Calcular_Costo(X_test, j, Y_test))
		#print("Costos_train: " + str(lista_costos_train))
		#print("Costos_test: " + str(lista_costos_test))
		Graficar_Costo_2(lista_costos_train, lista_costos_test, "../Resultados/Exp4/Data-" + str(i) + "-Costos.png")

def Experimento_4_separados(train_dataset, test_dataset, learning_rate):
	try:
		np_arr = Leer_Datos(train_dataset)
		np_arr_2 = Leer_Datos(test_dataset)
	except:
		print("Error: Los nombres de los datasets son incorrectos")
		return
	np_arr, Media, Desviacion = Normalizar_Datos(np_arr)
	np_arr = np_arr + 1
	np_arr_2 = Normalizar_Datos_MD(np_arr_2, Media, Desviacion)
	np_arr_2 = np_arr_2 + 1
	(X_train, Y_train), (X_test, Y_test) = separador(np_arr), separador(np_arr_2)
	theta = init_theta(X_train)
	theta, lista_thetas, lista_costos_train = Gradiente_Descendiente(X_train, theta, Y_train, learning_rate = learning_rate, interval = 50)
	lista_costos_test = []
	for j in lista_thetas:
		lista_costos_test.append(Calcular_Costo(X_test, j, Y_test))
	Graficar_Costo_2(lista_costos_train, lista_costos_test, "../Resultados/Exp4/Data-Other-Costos.png")
