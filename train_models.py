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
carpeta_config = os.path.dirname(os.path.abspath(args.config))#las rutas relativas parten de aquí


def entrenar(modelo, nombre, X, y=None):
    """Entrena el modelo y lo guarda en la ruta de su sección de train.conf."""
    ruta = os.path.join(carpeta_config, config[nombre]["ruta_modelo"])
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    modelo.fit(X, y).save(ruta)
    print(f"{nombre} guardado en {ruta}")


#keep_default_na=False: que un título o tag como "NA" no se lea como nulo
df = pd.read_csv(os.path.join(carpeta_config, config["datos"]["ruta_datos"]), keep_default_na=False)
df_likes = df[df["ratings_disabled"] == False]#con los ratings desactivados los likes valen siempre 0

entrenar(RFClassifier(), "rf_classifier", df, df["category"])
entrenar(RFRegressor(), "rf_regressor", df_likes, df_likes["likes"])
entrenar(KMeansClustering(), "kmeans_clustering", df)
entrenar(KNNRecommender(), "knn_recommender", df)
