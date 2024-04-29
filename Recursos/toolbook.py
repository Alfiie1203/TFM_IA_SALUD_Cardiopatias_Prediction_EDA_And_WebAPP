import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import numpy as np
import sys
from tabulate import tabulate
import os

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


def handle_duplicate_data(dataset):
    
    sys.stdout.flush()  # Forzar la salida inmediata de los mensajes
    duplicated_groups = dataset[dataset.duplicated(keep=False)].groupby(list(dataset.columns)).groups

    if not duplicated_groups:
        print("No hay valores duplicados en el conjunto de datos.")
        return dataset

    print("\nAhora, vamos a manejar los valores duplicados")

    # Eliminar duplicados
    print("\nOpciones para manejar datos duplicados:")
    print("1. Eliminar duplicados")
    print("2. Revisar las filas duplicadas")
    opcion = input("Por favor, seleccione una opción para manejar los datos duplicados (1-2) o presione 'enter' para continuar: ")

    if opcion == "1":
        datos_previos = dataset.copy()
        dataset = dataset.drop_duplicates()
        print("\nSe eliminaron los valores duplicados.")
        print("\nDatos duplicados eliminados:")
        print_data(datos_previos[datos_previos.duplicated()], "keys", "")
        print("\nDatos actualizados (duplicados eliminados):")
        print_data(dataset, "keys", "")
    elif opcion == "2":
        print("Valores duplicados agrupados:")
        for i, (group, indices) in enumerate(duplicated_groups.items(), start=1):
            print(f"\nGrupo {i}:")
            print_data(dataset.loc[indices], "keys", "")
        sys.stdout.flush()  # Forzar la salida inmediata de los mensajes
        handle_duplicate_data(dataset)
    elif opcion == "":
        pass
    else:
        print("Opción no válida. No se realizará ninguna acción.")

    return dataset

def handle_outliers(dataset):
    """
    Identifica y maneja los valores atípicos en un conjunto de datos.

    Args:
    - dataset: DataFrame de pandas, el conjunto de datos a procesar.

    Returns:
    - DataFrame: El conjunto de datos actualizado después de manejar los valores atípicos.
    """

    def detect_outliers(data, threshold=3):
        outliers = []
        mean = np.mean(data)
        std = np.std(data)
        for value in data:
            z_score = (value - mean) / std
            if np.abs(z_score) > threshold:
                outliers.append(value)
        return outliers

    sys.stdout.flush()  # Forzar la salida inmediata de los mensajes

    for columna in dataset.columns:
        sys.stdout.flush()  # Forzar la salida inmediata de los mensajes
        outliers = detect_outliers(dataset[columna])
        if outliers:
            print("\n_______________________________________")
            print(f"Los valores atípicos en la columna '{columna}' son:", outliers)
            lower_bound = np.min([value for value in dataset[columna] if value not in outliers])
            upper_bound = np.max([value for value in dataset[columna] if value not in outliers])
            print(f"Los valores de esta columna deberían estar en este rango: [{lower_bound}, {upper_bound}]")
            print("_______________________________________")

            print(f"\nOpciones para manejar valores atípicos en la columna '{columna}':")
            print("1. Imputar la media de la columna")
            print("2. Imputar la mediana de la columna")
            print("3. Imputar el valor mínimo de la columna")
            print("4. Imputar el valor máximo de la columna")
            print("5. Eliminar filas con valores atípicos en esta columna")
            print("6. Sustituir los valores atípicos con un valor específico")
            print("7. Mantener los valores atípicos intactos")
            sys.stdout.flush()  # Forzar la salida inmediata de los mensajes
            opcion = input("Por favor, seleccione una opción para esta columna (1-7) o presione 'enter' para continuar: ")

            if opcion == "1":
                sys.stdout.flush()  # Forzar la salida inmediata de los mensajes
                dataset[columna] = dataset[columna].fillna(dataset[columna].mean())
                print("\nSe imputa la media de la columna")
            elif opcion == "2":
                sys.stdout.flush()  # Forzar la salida inmediata de los mensajes
                dataset[columna] = dataset[columna].fillna(dataset[columna].median())
                print("\nSe imputa la mediana de la columna")
            elif opcion == "3":
                sys.stdout.flush()  # Forzar la salida inmediata de los mensajes
                dataset[columna] = dataset[columna].fillna(lower_bound)
                print("\nSe imputa el valor mínimo de la columna")
            elif opcion == "4":
                sys.stdout.flush()  # Forzar la salida inmediata de los mensajes
                dataset[columna] = dataset[columna].fillna(upper_bound)
                print("\nSe imputa el valor máximo de la columna")
            elif opcion == "5":
                sys.stdout.flush()  # Forzar la salida inmediata de los mensajes
                dataset = dataset[~dataset[columna].isin(outliers)]
                print("\nSe elimina filas con valores atípicos en esta columna")
            elif opcion == "6":
                sys.stdout.flush()  # Forzar la salida inmediata de los mensajes
                valor_especifico = input("Por favor, ingrese el valor específico con el que desea reemplazar los valores atípicos: ")
                dataset[columna] = dataset[columna].replace(outliers, valor_especifico)
                print("\nSe sustiye los valores atípicos con un valor específico")
            elif opcion == "7":
                sys.stdout.flush()  # Forzar la salida inmediata de los mensajes                
                print("\nSe mantiene los valores atípicos intactos")
                continue
            elif opcion == "":
                continue
            else:
                print("Opción no válida. No se realizará ninguna acción para esta columna.")
    return dataset


def correlationAnalysis(dataset):
    """
    Realiza un análisis de correlación para identificar las relaciones lineales entre las variables.

    Parameters:
    - dataset: DataFrame de pandas, conjunto de datos a analizar.
    """
    # Calcula la matriz de correlación
    correlation_matrix = dataset.corr()

    # Imprime la matriz de correlación de forma tabulada
    print("\nMatriz de correlación:")
    print(tabulate(correlation_matrix, headers='keys', tablefmt='fancy_grid'))
    
    # Visualiza la matriz de correlación
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Matriz de Correlación')
    plt.show()



def analyze_variable(dataset, target_name):
    """
    Analiza cada variable del conjunto de datos en función de la variable objetivo.

    Parameters:
    - dataset: DataFrame de pandas, conjunto de datos a analizar.
    - target_name: str, nombre de la variable objetivo.

    Returns:
    - dict: Diccionario donde las claves son los nombres de las variables y los valores son los resultados del análisis.
    """

    results = {}

    # Obtener los nombres de todas las columnas excepto la variable objetivo
    variable_names = [col for col in dataset.columns if col != target_name]

    for variable_name in variable_names:
        # Eliminar filas con valores NaN en las columnas de interés
        dataset_cleaned = dataset[[variable_name, target_name]].dropna()

        # Calcular la cantidad total de veces que aparece cada valor único de la variable
        counts = dataset_cleaned[variable_name].value_counts()

        # Calcular la cantidad total de veces que aparece cada valor único en porcentaje
        counts_percentage = counts / len(dataset_cleaned) * 100

        # Crear un DataFrame para almacenar los resultados
        results[variable_name] = pd.DataFrame({variable_name: counts.index, 'Count': counts.values, 'Percentage': counts_percentage.values})

        # Calcular la cantidad total de veces que aparece cada valor único de la variable objetivo
        target_counts = dataset_cleaned.groupby(variable_name)[target_name].value_counts().unstack().fillna(0)

        # Calcular la cantidad total de veces que aparece cada valor único de la variable objetivo en porcentaje
        target_counts_percentage = (target_counts.div(target_counts.sum(axis=1), axis=0) * 100).fillna(0)

        # Unir los resultados al DataFrame principal
        results[variable_name] = results[variable_name].join(target_counts, on=variable_name)
        results[variable_name] = results[variable_name].join(target_counts_percentage, on=variable_name, rsuffix='_percentage')

        # Renombrar las columnas
        results[variable_name] = results[variable_name].rename(columns={0: f'{target_name}_0', 1: f'{target_name}_1',
                                          0.0: f'{target_name}_0_percentage', 1.0: f'{target_name}_1_percentage'})

        # Crear un gráfico de barras para mostrar la cantidad de veces que la variable objetivo es "1" y "0" para cada valor único de la variable
        target_counts.plot(kind='bar', stacked=True)
        plt.title(f'Distribución de {target_name} para cada valor único de {variable_name}')
        plt.xlabel(variable_name)
        plt.ylabel(f'Cantidad de {target_name}')
        plt.legend(title=target_name)
        plt.show()

        # Imprimir la tabla de resultados
        print(f"Tabla de resultados para la variable '{variable_name}':")
        print(tabulate(results[variable_name], headers='keys', tablefmt='grid'))
        print()
