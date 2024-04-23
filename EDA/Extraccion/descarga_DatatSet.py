#Instalación de rcursos necesarios: pip install -r requirements.txt
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Obtener el conjunto de datos de enfermedades cardíacas
heart_disease = fetch_ucirepo(id=45)

# Convertir los datos y etiquetas a DataFrames de pandas
X = pd.DataFrame(data=heart_disease.data.features, columns=heart_disease.variables)
y = pd.DataFrame(data=heart_disease.data.targets, columns=['target'])

# Concatenar características y etiquetas en un solo DataFrame si es necesario
heart_data = pd.concat([X, y], axis=1)

# Exportar el DataFrame a un archivo CSV
heart_data.to_csv('../../Datos/Iniciales/heart_disease_dataset.csv', index=False)
