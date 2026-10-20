import argparse

import pandas as pd

from models.gmm_clustering import GMMClustering
from models.knn_recommender import KNNRecommender
from models.rf_classifier import RFClassifier
from models.rf_regressor import RFRegressor

#Ejemplo de uso: python inference_model.py rf_classifier pesos/rf_classifier.pkl nuevos_videos.csv

MODELOS = {
    "rf_classifier": RFClassifier,#predice la categoría
    "rf_regressor": RFRegressor,#predice los likes
    "gmm_clustering": GMMClustering,#asigna el grupo
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
    #el recomendador recibe títulos: recomendamos para cada título de la columna title
    for titulo, recomendaciones in zip(datos["title"], modelo.predict(list(datos["title"]))):
        print(f"\nRecomendaciones para: {titulo}")
        if recomendaciones is not None:
            print(recomendaciones.to_string(index=False))
else:
    resultado = datos.copy()
    resultado["prediccion"] = modelo.predict(datos)
    print(resultado.to_string(index=False))
