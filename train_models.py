import argparse
import configparser
import os

import pandas as pd

from models.kmeans_clustering import KMeansClustering
from models.knn_recommender import KNNRecommender
from models.rf_classifier import RFClassifier
from models.rf_regressor import RFRegressor

#Ejemplo de uso: python train_models.py train.conf

parser = argparse.ArgumentParser(description="Entrena los modelos y los guarda en disco.")
parser.add_argument("config", help="ruta del fichero de configuración (train.conf)")
args = parser.parse_args()

config = configparser.ConfigParser()
if not config.read(args.config, encoding="utf-8"):
    raise FileNotFoundError(f"No se encuentra el fichero de configuración: {args.config}")

carpeta_config = os.path.dirname(os.path.abspath(args.config))


def ruta(seccion, clave):
    """Devuelve la ruta del fichero de configuración; si es relativa, la toma desde la carpeta de train.conf."""
    return os.path.join(carpeta_config, config[seccion][clave])


def guardar(modelo, seccion):
    ruta_modelo = ruta(seccion, "ruta_modelo")
    os.makedirs(os.path.dirname(ruta_modelo), exist_ok=True)#creamos la carpeta si no existe
    modelo.save(ruta_modelo)
    print(f"{seccion} guardado en {ruta_modelo}")


df = pd.read_csv(ruta("datos", "ruta_datos"), keep_default_na=False)#keep_default_na=False: que un título o tag como "NA" no se lea como nulo

#Predicción de la categoría
rf_classifier = RFClassifier()
rf_classifier.fit(df, df["category"])
guardar(rf_classifier, "rf_classifier")

#Predicción de likes: solo vídeos con los ratings activados, ya que si no los likes valen siempre 0
df_likes = df[df["ratings_disabled"] == False]
rf_regressor = RFRegressor()
rf_regressor.fit(df_likes, df_likes["likes"])
guardar(rf_regressor, "rf_regressor")

#Clusterización: con todos los vídeos, igual que en el notebook
kmeans_clustering = KMeansClustering()
kmeans_clustering.fit(df)
guardar(kmeans_clustering, "kmeans_clustering")

#Recomendador
knn_recommender = KNNRecommender()
knn_recommender.fit(df)
guardar(knn_recommender, "knn_recommender")
