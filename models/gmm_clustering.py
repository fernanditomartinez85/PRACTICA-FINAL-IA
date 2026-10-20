import pickle

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


class GMMClustering:
    """GMM con 3 grupos sobre las 2 dimensiones del autoencoder.

    Es el mejor modelo del apartado de clusterización (silhouette de 0,67). El autoencoder (un MLPRegressor que aprende a
    reconstruir sus propias variables) comprime cada vídeo en 2 valores y GMM los agrupa.
    """

    COLUMNAS = ["views", "likes", "dislikes", "comment_count", "dias_desde_publicacion", "tags", "title", "channel_title"]
    VARIABLES_CLUSTER = ["views", "likes", "dislikes", "comment_count", "dias_desde_publicacion", "n_tags", "longitud_titulo", "n_videos_canal"]

    def __init__(self, n_components=3, random_state=42):
        self.modelo = GaussianMixture(n_components=n_components, random_state=random_state)

        #Logaritmo y escalado para tratar los outliers, igual que en el resto de modelos
        self.preprocess = Pipeline([("log", FunctionTransformer(np.log1p)), ("scaler", StandardScaler())])

        #Autoencoder 8 -> 6 -> 2 -> 6 -> 8: la capa central de 2 neuronas es la reducción sobre la que agrupamos
        self.autoencoder = MLPRegressor(hidden_layer_sizes=(6, 2, 6), activation="tanh", max_iter=1000, random_state=random_state)

        self.videos_por_canal = None#se calcula en fit: número de vídeos de cada canal en los datos de entrenamiento

    def fit(self, X, y=None):
        """Entrena el modelo. X: DataFrame o numpy array con las columnas de COLUMNAS. No necesita y (no supervisado)."""
        X = self.__to_dataframe(X)
        self.videos_por_canal = X["channel_title"].value_counts()
        X_procesado = self.__preprocess_data(X, entrenar=True)
        self.modelo.fit(X_procesado)
        return self

    def predict(self, X):
        """Devuelve el grupo de uno o varios vídeos."""
        X_procesado = self.__preprocess_data(self.__to_dataframe(X), entrenar=False)
        return self.modelo.predict(X_procesado)

    def save(self, ruta):
        """Guarda en disco (pickle) el estado de la clase: modelo, preprocess, autoencoder y vídeos por canal."""
        with open(ruta, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, ruta):
        """Carga de disco el estado guardado con save para poder predecir sin reentrenar."""
        with open(ruta, "rb") as f:
            self.__dict__.update(pickle.load(f))
        return self

    def __to_dataframe(self, X):
        """Convierte la entrada a DataFrame, tanto si es una sola observación como un conjunto."""
        if isinstance(X, pd.DataFrame):
            return X[self.COLUMNAS]
        if isinstance(X, pd.Series):#una sola observación
            return X.to_frame().T[self.COLUMNAS]
        if isinstance(X, dict):#una sola observación
            return pd.DataFrame([X])[self.COLUMNAS]
        X = np.array(X, dtype=object)
        if X.ndim == 1:#una sola observación
            X = X.reshape(1, -1)
        return pd.DataFrame(X, columns=self.COLUMNAS)#en un numpy array las columnas tienen que venir en el orden de COLUMNAS

    def __contar_tags(self, tags):
        if isinstance(tags, str):#los tags vienen del CSV separados por "|"
            tags = [tag for tag in tags.replace('"', "").split("|") if tag != ""]#quitamos las etiquetas vacías
        if list(tags) == ["[none]"]:#por si no hubiera ningun tag
            return 0
        return len(tags)

    def __codificar(self, X):
        """Pasa los vídeos por la mitad codificadora del autoencoder (8 -> 6 -> 2) y devuelve los 2 valores de la capa central."""
        capa = X
        for pesos, sesgos in zip(self.autoencoder.coefs_[:2], self.autoencoder.intercepts_[:2]):
            capa = np.tanh(capa @ pesos + sesgos)
        return capa

    def __preprocess_data(self, X, entrenar):
        """Crea n_tags, longitud_titulo y n_videos_canal, aplica logaritmo y escalado y reduce a 2 dimensiones con el autoencoder."""
        X = X.copy()
        X["n_tags"] = X["tags"].apply(self.__contar_tags)
        X["longitud_titulo"] = X["title"].str.len()
        X["n_videos_canal"] = X["channel_title"].map(self.videos_por_canal).fillna(1)#un canal que no estaba en el entrenamiento cuenta como 1 vídeo (el propio)
        X = X[self.VARIABLES_CLUSTER].astype(float)
        if entrenar:
            X_escalado = self.preprocess.fit_transform(X)
            self.autoencoder.fit(X_escalado, X_escalado)#aprende a reconstruir sus propias variables
        else:
            X_escalado = self.preprocess.transform(X)
        return self.__codificar(X_escalado)
