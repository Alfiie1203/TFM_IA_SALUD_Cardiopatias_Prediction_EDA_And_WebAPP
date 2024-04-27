import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

import sys
from tabulate import tabulate

def detailedExploration(dataset):
    """
    Realiza un análisis detallado del conjunto de datos.

    Parameters:
    - dataset: DataFrame de pandas, conjunto de datos a explorar.
    """
    # Muestra las primeras filas del DataFrame para verificar que los datos se han cargado correctamente:
    print("Muestra las primeras filas del DataFrame para verificar que los datos se han cargado correctamente:")
    print(tabulate(dataset.head(), headers='keys', tablefmt='fancy_grid'))

    # Muestra información general sobre el DataFrame, incluyendo el número de filas y columnas, nombres de columnas y tipos de datos
    print("\nInformación general sobre el DataFrame:")
    print(tabulate([["Número de Filas", dataset.shape[0]], ["Número de Columnas", dataset.shape[1]], ["Nombres de Columnas", ", ".join(dataset.columns)], ["Tipos de Datos", ", ".join(dataset.dtypes.astype(str))]], headers=['Característica', 'Valor'], tablefmt='fancy_grid'))

    # Muestra estadísticas descriptivas para las variables numéricas
    print("\nResumen estadístico y descriptivo del DataFrame:")
    print(tabulate(dataset.describe(), headers='keys', tablefmt='fancy_grid'))

    # Muestra la cantidad de valores únicos para cada columna
    print("\nValores únicos por columna:")
    print(tabulate([[col, dataset[col].nunique()] for col in dataset.columns], headers=['Columna', 'Valores Únicos'], tablefmt='fancy_grid'))


def allHistograms(dataset):

    print("\nGráficos para visualizar los datos:")

    # Histogramas para variables numéricas
    numeric_cols = dataset.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        plt.figure(figsize=(8, 6))
        sns.histplot(data=dataset, x=col, bins=20, kde=True)
        plt.title(f'Histograma de {col}')
        plt.xlabel(col)
        plt.ylabel('Frecuencia')
        plt.show()

    # Gráficos de barras para variables categóricas
    categorical_cols = dataset.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        plt.figure(figsize=(8, 6))
        sns.countplot(data=dataset, x=col, palette='viridis')
        plt.title(f'Gráfico de barras de {col}')
        plt.xlabel(col)
        plt.ylabel('Conteo')
        plt.xticks(rotation=45)
        plt.show()

def print_data(datos, mensaje, columna):
    sys.stdout.flush()  # Forzar la salida inmediata de los mensajes
    print(mensaje)
    if len(datos) > 15:
        opcion = input(f"Hay más de 15 valores en {columna}. ¿Desea imprimir todos los valores? (s/n): ")
        if opcion.lower() == "s":
            print(tabulate(datos, headers="keys", tablefmt="grid"))
        else:
            print(tabulate(datos.head(15), headers="keys", tablefmt="grid"))
            print(f"Se han mostrado los primeros 15 valores de {columna}.")
    else:
        print(tabulate(datos, headers="keys", tablefmt="grid"))

def handle_missing_data(dataset):
    columnas_con_null = dataset.columns[dataset.isnull().any()].tolist()

    if not columnas_con_null:
        print("No hay columnas con valores nulos.")
        return dataset

    print("_______________________________________")
    print("Cantidad de valores faltantes por columna:")
    print(tabulate(dataset[columnas_con_null].isna().sum().reset_index(), headers=["Columna", "Valores Faltantes"], tablefmt="grid"))
    print("Las columnas que no se muestren es porque no tienen valores nulos")
    print("_______________________________________")

    print("\nAhora, iteremos sobre las columnas que tienen nulos")

    for columna in columnas_con_null:
        sys.stdout.flush()  # Forzar la salida inmediata de los mensajes

        print("\nColumna:", columna)
        print("Los valores únicos de", columna, "y su frecuencia de aparición son:")
        print_data(pd.DataFrame(dataset[columna].value_counts()).reset_index(), "", columna)

        print("\nOpciones para manejar datos faltantes:")
        print("1. Eliminar filas con datos faltantes en esta columna")
        print("2. Imputar valores faltantes en esta columna (media, mediana, moda)")
        print("3. Interpolar valores faltantes en esta columna")
        print("4. Codificar datos faltantes en esta columna")
        print("5. Eliminar esta columna")
        opcion = input("Por favor, seleccione una opción para esta columna (1-5) o presione 'enter' para continuar: ")

        if opcion == "1":
            datos_previos = dataset.copy()
            dataset = dataset.dropna(subset=[columna])
            print("\nDatos eliminados en", columna, ":")
            print_data(datos_previos[datos_previos[columna].isna()], "keys", columna)
            print("\nSe eliminaron las filas con datos faltantes en esta columna")
            print("\nDatos actualizados en", columna, ":")
            print_data(dataset[dataset.index.isin(datos_previos[datos_previos[columna].isna()].index)], "keys", columna)
        elif opcion == "2":
            datos_previos = dataset.copy()
            imputar_opcion = input("¿Qué valor desea utilizar para imputar (media/m, mediana/md, moda/mo)? ")
            if imputar_opcion.lower() == "media" or imputar_opcion.lower() == "m":
                dataset[columna] = dataset[columna].fillna(dataset[columna].mean())
                print("\nSe imputa valores faltantes en esta columna con la media")
            elif imputar_opcion.lower() == "mediana" or imputar_opcion.lower() == "md":
                dataset[columna] = dataset[columna].fillna(dataset[columna].median())                
                print("\nSe imputa valores faltantes en esta columna con la mediana)")
            elif imputar_opcion.lower() == "moda" or imputar_opcion.lower() == "mo":
                dataset[columna] = dataset[columna].fillna(dataset[columna].mode().iloc[0])
                print("\nSe imputa valores faltantes en esta columna con la moda")
            else:
                print("Opción no válida. No se realizará ninguna acción para esta columna.")
            print("\nDatos actualizados en", columna, ":")
            print_data(dataset[dataset.index.isin(datos_previos[datos_previos[columna].isna()].index)], "keys", columna)
        elif opcion == "3":
            datos_previos = dataset.copy()
            dataset[columna] = dataset[columna].interpolate()
            print("se acaba de interpolar valores faltantes en esta columna")
            print("\nDatos actualizados en", columna, ":")
            print_data(dataset[dataset.index.isin(datos_previos[datos_previos[columna].isna()].index)], "keys", columna)
        elif opcion == "4":
            datos_previos = dataset.copy()
            valor_codificar = input("¿Qué valor desea utilizar para codificar los datos faltantes en esta columna? ")
            dataset[columna] = dataset[columna].fillna(valor_codificar)
            print("Se codificaron los datos faltantes en esta columna con:", valor_codificar)
            print("\nDatos actualizados en", columna, ":")
            print_data(dataset[dataset.index.isin(datos_previos[datos_previos[columna].isna()].index)], "keys", columna)
        elif opcion == "5":
            datos_previos = dataset.copy()
            dataset = dataset.drop(columna, axis=1)
            print("\nColumna eliminada:", columna)
            print("\nDatos eliminados en", columna, ":")
            print_data(datos_previos[datos_previos[columna].isna()], "keys", columna)
            print("\nDatos actualizados (columna eliminada):")
            print_data(dataset, "keys", columna)
        elif opcion == "":
            continue
        else:
            print("Opción no válida. No se realizará ninguna acción para esta columna.")

    return dataset