import argparse
import sys

import pandas as pd

from models.kmeans_clustering import KMeansClustering
from models.knn_recommender import KNNRecommender
from models.rf_classifier import RFClassifier
from models.rf_regressor import RFRegressor

#Ejemplo de uso: python inference_model.py rf_classifier rf_classifier.pkl nuevos_videos.csv

#hay títulos en ruso, coreano, japonés...: sin esto, en Windows, guardar la salida en un fichero da UnicodeEncodeError
sys.stdout.reconfigure(encoding="utf-8")

modelos = {"rf_classifier": RFClassifier,#categoría
           "rf_regressor": RFRegressor,#likes
           "kmeans_clustering": KMeansClustering,#cluster
           "knn_recommender": KNNRecommender}#vídeos recomendados

parser = argparse.ArgumentParser(description="Carga un modelo entrenado y realiza la inferencia de nuevos datos.")
parser.add_argument("modelo", choices=modelos.keys(), help="tipo de modelo a utilizar")
parser.add_argument("pesos", help="ruta del fichero pickle con el modelo entrenado")
parser.add_argument("datos", help="fichero .csv o .pkl con las mismas columnas que los datos limpios")
args = parser.parse_args()

#keep_default_na=False: que un título o tag como "NA" no se lea como nulo
datos = pd.read_pickle(args.datos) if args.datos.endswith(".pkl") else pd.read_csv(args.datos, keep_default_na=False)
modelo = modelos[args.modelo]().load(args.pesos)

if args.modelo == "knn_recommender":
    for (_, fila), recomendaciones in zip(datos.iterrows(), modelo.predict(datos)):
        indicado = " | ".join(f"{col}: {fila[col]}" for col in KNNRecommender.busqueda if fila.get(col, "") != "")
        print(f"\nRecomendaciones para: {indicado}")
        if recomendaciones is not None:
            print(recomendaciones.to_string(index=False))
else:
    datos["prediccion"] = modelo.predict(datos)
    print(datos.to_string(index=False))
