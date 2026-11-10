import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


class KMeansClustering:

    def __init__(self):
        
        logaritmo = FunctionTransformer(np.log1p)
        scaler = StandardScaler()
        #Definimos las transformaciones numéricas para tratar los outliers
        self.pipeline_num = Pipeline([("log", logaritmo),("scaler", scaler)])
        self.pca = PCA(n_components=2, random_state=42)
        self.kmeans = KMeans(n_clusters=2, random_state=42)

    def __preprocess_data(self, df, entrenar):
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(np.array(df, dtype=object).reshape(-1, 8),#valores en filas de 8 columnas
                              columns=["views", "likes", "dislikes", "comment_count", "dias_desde_publicacion",
                                       "tags", "title", "channel_title"])
            
        #copia para no añadir columnas al df que nos pasan
        df = df.copy()

        #creamos n_tags para que cuente los tags y ayudar a clusterizar.
        df["n_tags"] = df["tags"].apply(self.__contar_tags)

        df["longitud_titulo"] = df["title"].str.len()

        if entrenar:#vídeos de cada canal en los datos de entrenamiento
            self.videos_por_canal = df["channel_title"].value_counts()

        #¿el canal de este vídeo sale mucho en tendencias o casi nunca? Un canal que no estaba al
        #entrenar cuenta como 1 vídeo (el propio)
        df["n_videos_canal"] = df["channel_title"].map(self.videos_por_canal).fillna(1)

        variables_cluster = ["views", "likes", "dislikes", "comment_count", "dias_desde_publicacion",
                             "n_tags", "longitud_titulo", "n_videos_canal"]
        #astype: con un numpy array los números llegan como objeto y el logaritmo daría error
        X_cluster = df[variables_cluster].astype(float)

        if not entrenar:#al predecir
            X_cluster = self.pipeline_num.transform(X_cluster)
            X_pca = self.pca.transform(X_cluster)
            return X_pca
        
        X_cluster = self.pipeline_num.fit_transform(X_cluster)
        X_pca = self.pca.fit_transform(X_cluster)
        return X_pca

    def __contar_tags(self, tags):
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
        return etiquetas_kmeans

    def save(self, ruta):
        """Guarda en un pickle todo el estado del modelo (modelo entrenado y transformaciones)."""
        with open(ruta, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, ruta):
        """Recupera el estado guardado con save, para predecir sin reentrenar."""
        with open(ruta, "rb") as f:
            self.__dict__.update(pickle.load(f))
        return self
