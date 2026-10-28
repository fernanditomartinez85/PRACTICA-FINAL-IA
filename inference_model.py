import argparse
import sys

import pandas as pd

from models.kmeans_clustering import KMeansClustering
from models.knn_recommender import KNNRecommender
from models.rf_classifier import RFClassifier
from models.rf_regressor import RFRegressor

#Ejemplo de uso: python inference_model.py rf_classifier pesos/rf_classifier.pkl nuevos_videos.csv

#hay títulos en ruso, coreano, japonés...: sin esto, en Windows, al guardar la salida en un fichero
#(python inference_model.py ... > salida.txt) da UnicodeEncodeError
sys.stdout.reconfigure(encoding="utf-8")

MODELOS = {
    "rf_classifier": RFClassifier,#predice la categoría
    "rf_regressor": RFRegressor,#predice los likes
    "kmeans_clustering": KMeansClustering,#asigna el cluster
    "knn_recommender": KNNRecommender#recomienda vídeos parecidos a los títulos del fichero
}

parser = argparse.ArgumentParser(description="Carga un modelo entrenado y realiza la inferencia de nuevos datos.")
parser.add_argument("modelo", choices=MODELOS.keys(), help="tipo de modelo a utilizar")
parser.add_argument("pesos", help="ruta del fichero pickle con el modelo entrenado")
parser.add_argument("datos", help="fichero de datos de entrada (.csv o .pkl) con las mismas columnas que los datos limpios")
args = parser.parse_args()

if args.datos.endswith(".pkl"):
    datos = pd.read_pickle(args.datos)
else:
    datos = pd.read_csv(args.datos, keep_default_na=False)#keep_default_na=False: que un título o tag como "NA" no se lea como nulo

modelo = MODELOS[args.modelo]()
modelo.load(args.pesos)

if args.modelo == "knn_recommender":
    #el recomendador recibe el título, el canal o la categoría de cada vídeo (basta con uno de los tres)
    columnas = [c for c in ["title", "channel_title", "category"] if c in datos.columns]
    for (_, fila), recomendaciones in zip(datos.iterrows(), modelo.predict(datos)):
        indicado = " | ".join(f"{c}: {fila[c]}" for c in columnas if fila[c] != "")
        print(f"\nRecomendaciones para: {indicado}")
        if recomendaciones is not None:
            print(recomendaciones.to_string(index=False))
else:
    resultado = datos.copy()
    resultado["prediccion"] = modelo.predict(datos)
    print(resultado.to_string(index=False))
