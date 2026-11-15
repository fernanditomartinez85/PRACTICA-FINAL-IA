import argparse
import configparser
import pandas as pd
from models.kmeans_clustering import KMeansClustering
from models.knn_recommender import KNNRecommender
from models.rf_classifier import RFClassifier
from models.rf_regressor import RFRegressor

#Ejemplo de uso: python train_models.py train.conf

parser = argparse.ArgumentParser(description="Entrena los modelos y los guarda en disco.")
parser.add_argument("ruta_train_conf", help="Ruta del fichero de configuración (train.conf)")
args = parser.parse_args()

config = configparser.ConfigParser()
lectura_train_conf = config.read(args.ruta_train_conf, encoding="utf-8")
if not lectura_train_conf:
    raise FileNotFoundError(f"No se encuentra el fichero de configuración: {args.ruta_train_conf}")

#keep_default_na=False: que un título o tag como "NA" no se lea como nulo
df = pd.read_csv(config["datos"]["ruta_datos"], keep_default_na=False)
df_likes = df[df["ratings_disabled"] == False]#con los ratings desactivados los likes valen siempre 0, lo usamos en regressor


def entrenar(modelo, seccion_train_conf, X, y=None):#y=None porque hay modelos que no la necesitan
    """Entrena el modelo y lo guarda en la ruta de su sección de train.conf."""
    ruta = config[seccion_train_conf]["ruta_modelo"]
    modelo.fit(X, y)
    #entrena cada modelo llamando a fit "con el conjunto de datos", sin pedir ninguna división.
    modelo.save(ruta)
    print(f"{seccion_train_conf} guardado en {ruta}")

entrenar(RFClassifier(), "rf_classifier", df, df["category"])
entrenar(RFRegressor(), "rf_regressor", df_likes, df_likes["likes"])
entrenar(KMeansClustering(), "kmeans_clustering", df)
entrenar(KNNRecommender(), "knn_recommender", df)
