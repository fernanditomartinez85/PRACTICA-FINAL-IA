import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


class KMeansClustering:

    def __init__(self):

        #en vez del one hot: logaritmo y escalado de las numéricas, y después PCA
        logaritmo = FunctionTransformer(np.log1p)
        scaler = StandardScaler()
        self.pipeline_num = Pipeline([("log", logaritmo),("scaler", scaler)])
        self.pca = PCA(n_components=2, random_state=42)
        self.kmeans = KMeans(n_clusters=2, random_state=42)

    def __preprocess_data(self, df, entrenar):#entrenar en vez de y: K-Means no tiene y
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(np.array(df, dtype=object).reshape(-1, 8),#valores en filas de 8 columnas
                              columns=["views", "likes", "dislikes", "comment_count", "dias_desde_publicacion",
                                       "tags", "title", "channel_title"])

        #copia: aquí se añaden columnas y así no cambiamos el df que nos pasan
        df = df.copy()

        #columnas nuevas que no tiene el clasificador: número de tags, longitud del título y vídeos del canal
        df["n_tags"] = df["tags"].apply(self.__contar_tags)

        df["longitud_titulo"] = df["title"].str.len()

        if entrenar:#vídeos de cada canal, se aprende al entrenar
            self.videos_por_canal = df["channel_title"].value_counts()

        #un canal que no estaba al entrenar cuenta como 1 vídeo (el propio)
        df["n_videos_canal"] = df["channel_title"].map(self.videos_por_canal).fillna(1)

        variables_cluster = ["views", "likes", "dislikes", "comment_count", "dias_desde_publicacion",
                             "n_tags", "longitud_titulo", "n_videos_canal"]
        #astype: con un numpy array los números llegan como objeto y el logaritmo daría error
        X_cluster = df[variables_cluster].astype(float)

        if not entrenar:#al predecir
            X_cluster = self.pipeline_num.transform(X_cluster)
            X_pca = self.pca.transform(X_cluster)#segunda transformación: PCA
            return X_pca

        X_cluster = self.pipeline_num.fit_transform(X_cluster)
        X_pca = self.pca.fit_transform(X_cluster)#segunda transformación: PCA
        return X_pca#sin y: solo devuelve X

    def __contar_tags(self, tags):#método extra para la columna n_tags
        if isinstance(tags, str):#del csv vienen separados por "|"
            tags = [tag for tag in tags.replace('"', "").split("|") if tag != ""]
        return 0 if list(tags) == ["[none]"] else len(tags)

    def fit(self, df, y=None):#y=None: train_models.py llama a fit con dos argumentos, aunque aquí no hay y
        X_pca = self.__preprocess_data(df, entrenar=True)
        self.kmeans.fit(X_pca)
        return self

    def predict(self, df):
        X_pca = self.__preprocess_data(df, entrenar=False)
        etiquetas_kmeans = self.kmeans.predict(X_pca)
        return etiquetas_kmeans#sin inverse_transform: devuelve el número de grupo

    def save(self, ruta):
        with open(ruta, "wb") as f:
            pickle.dump(self, f)

    def load(self, ruta):
        with open(ruta, "rb") as f:
            modelo_entrenado = pickle.load(f)
        return modelo_entrenado
