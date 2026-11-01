import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from models.utils import a_dataframe, cargar, guardar


class KMeansClustering:
    """Agrupa los vídeos: K-Means con 2 grupos sobre 2 componentes de PCA (GridSearchCV, silhouette de 0,41)."""

    COLUMNAS = ["views", "likes", "dislikes", "comment_count", "dias_desde_publicacion", "tags", "title", "channel_title"]
    VARIABLES = ["views", "likes", "dislikes", "comment_count", "dias_desde_publicacion",
                 "n_tags", "longitud_titulo", "n_videos_canal"]

    def __init__(self, n_clusters=2, n_components=2, random_state=42):
        self.modelo = KMeans(n_clusters=n_clusters, random_state=random_state)
        #logaritmo y escalado para los outliers, y PCA a 2 componentes
        self.preprocess = Pipeline([("log", FunctionTransformer(np.log1p)),
                                    ("scaler", StandardScaler()),
                                    ("pca", PCA(n_components=n_components, random_state=random_state))])
        self.videos_por_canal = None#se calcula en fit

    def fit(self, X, y=None):
        self.videos_por_canal = a_dataframe(X, self.COLUMNAS)["channel_title"].value_counts()
        self.modelo.fit(self.__preprocess_data(X, entrenar=True))
        return self

    def predict(self, X):
        return self.modelo.predict(self.__preprocess_data(X, entrenar=False))

    def save(self, ruta):
        guardar(self, ruta)

    def load(self, ruta):
        return cargar(self, ruta)

    def __contar_tags(self, tags):
        if isinstance(tags, str):#del CSV vienen separados por "|"
            tags = [tag for tag in tags.replace('"', "").split("|") if tag != ""]
        return 0 if list(tags) == ["[none]"] else len(tags)

    def __preprocess_data(self, X, entrenar):
        """Crea n_tags, longitud_titulo y n_videos_canal, y aplica logaritmo, escalado y PCA."""
        X = a_dataframe(X, self.COLUMNAS).copy()
        X["n_tags"] = X["tags"].apply(self.__contar_tags)
        X["longitud_titulo"] = X["title"].str.len()
        #un canal que no estaba al entrenar cuenta como 1 vídeo (el propio)
        X["n_videos_canal"] = X["channel_title"].map(self.videos_por_canal).fillna(1)
        X = X[self.VARIABLES].astype(float)
        return self.preprocess.fit_transform(X) if entrenar else self.preprocess.transform(X)
