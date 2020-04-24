from Intermediarios import *
import json

def main():
    opcion = 1
    learning_rates = [0.01, 0.05, 0.1, 0.2, 0.4]
    fixed_datasets = ["../Datos/ex1data2(Home_1f).csv", "../Datos/oceano_simple.csv", "../Datos/petrol_consumption.csv"]
    while opcion <= 4 and opcion > 0:
        print("Seleccione el experimento que desea realizar:")
        print("Experimento 1: Mostrar error cuadratico usando Ec. normal de un dataset")
        print("Experimento 2: Buscar mejores parametros de entrenamiento de un dataset")
        print("Experimento 3: Plotear Precio viviendas")
        print("Experimento 4: Plotear costo de un dataset")
        opcion = check_type(int)
        #En los experimentos
        if opcion == 1:
            print("Seleccione el Dataset: ")
            print("Opcion 1: Datasets principales")
            print("Otro numero: Dataset personalizado en la carpeta ../Otros_Datos/")
            opcion2 = check_type(int)
            print("Ingrese el porcentaje de la muestra para considerar en la Ecuacion Normal Max: 1, min: 0.01")
            sample_test = check_type(float)
            if opcion2 == 1:
                buffer1 = Experimento_1(fixed_datasets, sample_test)
            else:
                print("Ingrese solo el nombre del archivo: (en la ubicacion ../Otros_Datos/)")
                filename = input()
                buffer1 = Experimento_1(["../Otros_Datos/" + filename], sample_test)
            print("Resultado:")
            print(str(buffer1))
            with open('../Resultados/Exp1/Resultado1.txt', 'w') as f:
                f.write(str(buffer1))
            print("Resultados guardados en '../Resultados/Exp1'")
        elif opcion == 2:
            print("Seleccione el Dataset: ")
            print("Opcion 1: Datasets principales")
            print("Otro numero: Dataset personalizado en la carpeta ../Otros_Datos/")
            opcion2 = check_type(int)
            if opcion2 == 1:
                statistics = Experimento_2(fixed_datasets, learning_rates)
            else:
                print("Ingrese solo el nombre del dataset de entrenamiento: (en la ubicacion ../Otros_Datos/)")
                train_dataset_name = input()
                print("Ingrese solo el nombre del dataset de prueba: (en la ubicacion ../Otros_Datos/)")
                test_dataset_name = input()
                statistics = Experimento_2_separados("../Otros_Datos/" + train_dataset_name, "../Otros_Datos/" + test_dataset_name, + learning_rates)
            print("Resultado:")
            print(statistics)
            with open('../Resultados/Exp2/Resultado2.json', 'w') as json_file:
                json.dump(statistics, json_file)
            print("Resultados guardados en '../Resultados/Exp2")
        elif opcion == 3:
            print("Seleccione el learning rate")
            learning_rate = check_type(float)
            print("Seleccione las iteraciones:")
            iteraciones = check_type(int)
            print("Seleccione el Dataset:")
            print("Opcion 1: Seleccionar '../Datos/ex1data2(Home_1f).csv' como dataset principal")
            print("Otro: Datasets separados en '../Otros_Datos/")
            opcion2 = check_type(int)
            if opcion2 == 1:
                Experimento_3(learning_rate = learning_rate, iteraciones = iteraciones)
            else:
                print("Ingrese solo el nombre del dataset de entrenamiento: (en la ubicacion ../Otros_Datos/)")
                train_dataset_name = input()
                print("Ingrese solo el nombre del dataset de prueba: (en la ubicacion ../Otros_Datos/)")
                test_dataset_name = input()
                Experimento_3_separados("../Otros_Datos/" + train_dataset_name, "../Otros_Datos/" + test_dataset_name, learning_rate, iteraciones)
            print("Resultados guardados en '../Resultados/Exp3'")
        elif opcion == 4:
            print("Seleccione el learning rate")
            learning_rate = check_type(float)
            print("Resultados guardados en '../Resultados/Exp4'")
            print("Seleccione los Datasets:")
            print("Opcion 1: Dataset de la carpeta '../Datos/'")
            print("Otro: Otro dataset de la carpeta '../Otros_Datos/'")
            opcion2 = check_type(int)
            if opcion2 == 1:
                Experimento_4(fixed_datasets, learning_rate)
            else:
                print("Ingrese solo el nombre del dataset de entrenamiento: (en la ubicacion ../Otros_Datos/)")
                train_dataset_name = input()
                print("Ingrese solo el nombre del dataset de prueba: (en la ubicacion ../Otros_Datos/)")
                test_dataset_name = input("../Otros_Datos/" + train_dataset_name, "../Otros_Datos/" + test_dataset_name, learning_rate)
                Experimento_4_separados()
            print("Resultados guardados en '../Resultados/Exp4'")
        else:
            print("fin del programa")

if __name__ == "__main__":
    main()
