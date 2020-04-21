from Intermediarios import *

def main():
    opcion = 1
    learning_rates = [0.01, 0.05, 0.1, 0.2, 0.4]
    fixed_datasets = ["../Datos/ex1data2(Home_1f).csv", "../Datos/oceano_simple.csv", "../Datos/petrol_consumption.csv"]
    while opcion <= 4 and opcion > 0:
        print("Seleccione el experimento que desea realizar:")
        print("Experimeto 1: Mostrar error cuadratico usando Ec. normal de un dataset")
        print("Experimeto 2: Buscar mejores parametros de entrenamiento de un dataset")
        print("Experimeto 3: Plotear Precio viviendas")
        print("Experimeto 4: Plotear petrol_consumptionto de un dataset")
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
                Experimeto_1(fixed_datasets, sample_test)
            else:
                print("Ingrese solo el nombre del archivo")
                filename = input()
                Experimeto_1(["../Otros_Datos/" + filename], sample_test)
        elif opcion == 2:

        elif opcion == 3:

        elif opcion == 4:

        else:
            print("fin del programa")

if __name__ == "__main__":
    main()
