import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


class KMeansClustering:
    """K-Means con 4 clusters sobre las 2 componentes de PCA.

    Es el mejor modelo del apartado de clusterización (silhouette de 0,35).
    """

    COLUMNAS = ["views", "likes", "dislikes", "comment_count", "dias_desde_publicacion", "tags", "title", "channel_title"]
    VARIABLES_CLUSTER = ["views", "likes", "dislikes", "comment_count", "dias_desde_publicacion", "n_tags", "longitud_titulo", "n_videos_canal"]

    def __init__(self, n_clusters=4, n_components=2, random_state=42):
        self.modelo = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)

        #Logaritmo y escalado para tratar los outliers, y PCA para reducir a 2 componentes antes de clusterizar
        self.preprocess = Pipeline([("log", FunctionTransformer(np.log1p)),
                                    ("scaler", StandardScaler()),
                                    ("pca", PCA(n_components=n_components, random_state=random_state))])

        self.videos_por_canal = None#se calcula en fit: número de vídeos de cada canal en los datos de entrenamiento

    def fit(self, X, y=None):
        """Entrena el modelo. X: DataFrame o numpy array con las columnas de COLUMNAS. No necesita y (no supervisado)."""
        X = self.__to_dataframe(X)
        self.videos_por_canal = X["channel_title"].value_counts()
        X_procesado = self.__preprocess_data(X, entrenar=True)
        self.modelo.fit(X_procesado)
        return self

    def predict(self, X):
        """Devuelve el cluster de uno o varios vídeos."""
        X_procesado = self.__preprocess_data(self.__to_dataframe(X), entrenar=False)
        return self.modelo.predict(X_procesado)

    def save(self, ruta):
        """Guarda en disco (pickle) el estado de la clase: modelo, preprocess y vídeos por canal."""
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

    def __preprocess_data(self, X, entrenar):
        """Crea n_tags, longitud_titulo y n_videos_canal, y aplica logaritmo, escalado y PCA."""
        X = X.copy()
        X["n_tags"] = X["tags"].apply(self.__contar_tags)
        X["longitud_titulo"] = X["title"].str.len()
        X["n_videos_canal"] = X["channel_title"].map(self.videos_por_canal).fillna(1)#un canal que no estaba en el entrenamiento cuenta como 1 vídeo (el propio)
        X = X[self.VARIABLES_CLUSTER].astype(float)
        if entrenar:
            return self.preprocess.fit_transform(X)
        return self.preprocess.transform(X)
