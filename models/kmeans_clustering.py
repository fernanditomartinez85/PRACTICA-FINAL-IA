import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


class KMeansClustering:

    columnas = ["views", "likes", "dislikes", "comment_count", "dias_desde_publicacion", "tags", "title", "channel_title"]
    variables = ["views", "likes", "dislikes", "comment_count", "dias_desde_publicacion",
                 "n_tags", "longitud_titulo", "n_videos_canal"]

    def __init__(self, n_clusters=2, n_components=2, random_state=42):
        self.modelo = KMeans(n_clusters=n_clusters, random_state=random_state)
        #logaritmo y escalado para los outliers, y pca a 2 componentes
        self.preprocess = Pipeline([("log", FunctionTransformer(np.log1p)),
                                    ("scaler", StandardScaler()),
                                    ("pca", PCA(n_components=n_components, random_state=random_state))])
        self.videos_por_canal = None#se calcula en fit

    def __preprocess_data(self, X, entrenar):
        """Crea n_tags, longitud_titulo y n_videos_canal, y aplica logaritmo, escalado y pca."""
        #un vídeo (diccionario o Series) o varios (DataFrame, o numpy array con las columnas en orden)
        if isinstance(X, (dict, pd.Series)):
            X = pd.DataFrame([X])
        elif not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.array(X, dtype=object).reshape(-1, len(self.columnas)), columns=self.columnas)
        X = X[self.columnas].copy()
        if entrenar:#vídeos de cada canal en los datos de entrenamiento
            self.videos_por_canal = X["channel_title"].value_counts()
        X["n_tags"] = X["tags"].apply(self.__contar_tags)
        X["longitud_titulo"] = X["title"].str.len()
        #un canal que no estaba al entrenar cuenta como 1 vídeo (el propio)
        X["n_videos_canal"] = X["channel_title"].map(self.videos_por_canal).fillna(1)
        X = X[self.variables].astype(float)
        return self.preprocess.fit_transform(X) if entrenar else self.preprocess.transform(X)

    def __contar_tags(self, tags):
        if isinstance(tags, str):#del csv vienen separados por "|"
            tags = [tag for tag in tags.replace('"', "").split("|") if tag != ""]
        return 0 if list(tags) == ["[none]"] else len(tags)

    def fit(self, X, y=None):
        self.modelo.fit(self.__preprocess_data(X, entrenar=True))
        return self

    def predict(self, X):
        return self.modelo.predict(self.__preprocess_data(X, entrenar=False))

    def save(self, ruta):
        """Guarda en un pickle todo el estado del modelo (modelo entrenado y transformaciones)."""
        with open(ruta, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, ruta):
        """Recupera el estado guardado con save, para predecir sin reentrenar."""
        with open(ruta, "rb") as f:
            self.__dict__.update(pickle.load(f))
        return self
