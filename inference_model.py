import argparse
import pandas as pd
from models.kmeans_clustering import KMeansClustering
from models.knn_recommender import KNNRecommender
from models.rf_classifier import RFClassifier
from models.rf_regressor import RFRegressor

#Ejemplo de uso: python inference_model.py rf_classifier rf_classifier.pkl datos_limpios.csv

modelos = {"rf_classifier": RFClassifier,
           "rf_regressor": RFRegressor,
           "kmeans_clustering": KMeansClustering,
           "knn_recommender": KNNRecommender}

parser = argparse.ArgumentParser(description="Carga un modelo entrenado y realiza la inferencia de nuevos datos.")
parser.add_argument("modelo", choices=modelos.keys(), help="tipo de modelo a utilizar")#si te equivocas choices te da un error y te dice las opciones válidas
parser.add_argument("pesos", help="ruta del fichero pickle con el modelo entrenado")
parser.add_argument("datos", help="fichero .csv con las mismas columnas que los datos limpios")
args = parser.parse_args()

#keep_default_na=False: que un título o tag como "NA" no se lea como nulo
datos = pd.read_csv(args.datos, keep_default_na=False)
modelo = modelos[args.modelo]().load(args.pesos)

if args.modelo == "knn_recommender":
    recomendaciones = modelo.predict(datos)
    for i in range(len(datos)):
        print(f"\nRecomendaciones para la fila {i + 1} del fichero:")
        if recomendaciones[i] is not None:
            print(recomendaciones[i].to_string(index=False))
else:
    datos["prediccion"] = modelo.predict(datos)
    print(datos.to_string(index=False))
