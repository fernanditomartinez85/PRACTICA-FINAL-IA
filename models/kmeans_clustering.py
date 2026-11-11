import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer, StandardScaler


class KMeansClustering:

    def __init__(self):

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2, random_state=42)
        self.kmeans = KMeans(n_clusters=2, random_state=42)

    def __preprocess_data(self, df):
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(np.array(df, dtype=object).reshape(-1, 8),#valores en filas de 8 columnas
                              columns=["views", "likes", "dislikes", "comment_count", "dias_desde_publicacion",
                                       "tags", "title", "channel_title"])

        df["tags"] = df["tags"].str.split("|")#del csv vienen como texto separado por "|"
        df["n_tags"] = df["tags"].apply(self.__contar_tags)

        df["longitud_titulo"] = df["title"].str.len()

        return df

    def __contar_tags(self, tags):
        if tags == ["[none]"]:#por si no hubiera ningun tag
            return 0
        return len(tags)

    def fit(self, df, y=None):#y=None: train_models.py llama a fit con dos argumentos, aunque aquí no hay y
        df = self.__preprocess_data(df)
        self.videos_por_canal = df["channel_title"].value_counts()
        df["n_videos_canal"] = df["channel_title"].map(self.videos_por_canal)
        #astype: con un numpy array los números llegan como objeto y el logaritmo daría error
        X_cluster = df[["views", "likes", "dislikes", "comment_count", "dias_desde_publicacion",
                        "n_tags", "longitud_titulo", "n_videos_canal"]].astype(float)
        logaritmo = FunctionTransformer(np.log1p)
        X_cluster = logaritmo.transform(X_cluster)
        X_cluster = self.scaler.fit_transform(X_cluster)
        X_pca = self.pca.fit_transform(X_cluster)
        self.kmeans.fit(X_pca)
        return self

    def predict(self, df):
        df = self.__preprocess_data(df)
        #un canal que no estaba al entrenar cuenta como 1 vídeo (el propio)
        df["n_videos_canal"] = df["channel_title"].map(self.videos_por_canal).fillna(1)
        X_cluster = df[["views", "likes", "dislikes", "comment_count", "dias_desde_publicacion",
                        "n_tags", "longitud_titulo", "n_videos_canal"]].astype(float)
        logaritmo = FunctionTransformer(np.log1p)
        X_cluster = logaritmo.transform(X_cluster)
        X_cluster = self.scaler.transform(X_cluster)
        X_pca = self.pca.transform(X_cluster)
        etiquetas_kmeans = self.kmeans.predict(X_pca)
        return etiquetas_kmeans#sin inverse_transform: devuelve el número de grupo

    def save(self, ruta):
        with open(ruta, "wb") as f:
            pickle.dump(self, f)

    def load(self, ruta):
        with open(ruta, "rb") as f:
            modelo_entrenado = pickle.load(f)
        return modelo_entrenado
