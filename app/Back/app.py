from flask import Flask, request, jsonify
from flask_cors import CORS  # Importa CORS desde la extensión flask_cors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import toolbook as toolbook

app = Flask(__name__)
CORS(app)  # Aplica CORS a todas las rutas de la aplicación

dataset = pd.read_csv('../../Datos/Transformaciones/heart_disease_dataset_no_num.csv')

X_train, X_test, y_train, y_test = toolbook.preprocess_data(dataset, 'target', 0.25, 1)

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

@app.route('/predecir', methods=['POST'])
def predecir():
    datos = request.json
    age = datos['age']
    sex = datos['sex']
    cp = datos['cp']

    # Convertir el JSON en DataFrame
    Newdataset = pd.DataFrame(datos, index=[0])

    # Predicción de probabilidades para el nuevo dataset
    y_new_prob = classifier.predict_proba(Newdataset)

    # Obtener las probabilidades de predicción para las clases 0 y 1
    prob_presencia = y_new_prob[0][1]

    # Calcular los porcentajes de presencia
    porcentaje_presencia = prob_presencia * 100


    resultado = age + "," + sex + "," + cp

    return jsonify({'resultado': str(porcentaje_presencia)})

if __name__ == '__main__':
    app.run(debug=True)
