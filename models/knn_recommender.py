import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier


class KNNRecommender:
    """Recomendador basado en contenido: TF-IDF sobre título, tags, canal y categoría, y KNN con la distancia del coseno.

    Es el modelo del apartado del recomendador (72 % de recomendaciones de la misma categoría frente al 15 % al azar).
    """

    COLUMNAS = ["title", "tags", "channel_title", "category", "pais", "views"]

    def __init__(self, n_neighbors=11, max_features=50000, min_df=2):
        #n_neighbors=11: el vecino más cercano de un vídeo es el propio vídeo, así que pedimos 11 para quedarnos con 10
        self.modelo = KNeighborsClassifier(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
        self.preprocess = TfidfVectorizer(stop_words="english", min_df=min_df, max_features=max_features)
        self.videos = None#se guardan en fit: son los vídeos entre los que se recomienda
        self.X_tfidf = None

    def fit(self, X, y=None):
        """Entrena el recomendador. X: DataFrame o numpy array con las columnas de COLUMNAS (no necesita y)."""
        videos = self.__to_dataframe(X)
        videos = videos.sort_values(by="views", ascending=False)
        videos = videos.drop_duplicates(subset=["title", "channel_title"], keep="first")#un único registro por vídeo para no recomendar el mismo vídeo varias veces
        self.videos = videos.reset_index(drop=True)#cada fila coincide con su posición en la matriz TF-IDF

        self.X_tfidf = self.__preprocess_data(self.videos, entrenar=True)
        self.modelo.fit(self.X_tfidf, self.videos["category"])#el fit necesita etiqueta, aunque para recomendar solo usamos kneighbors
        return self

    def predict(self, X, n=10):
        """Recibe uno o varios textos de título y devuelve los n vídeos más parecidos al vídeo más visto que lo contiene.

        Con un solo texto devuelve un DataFrame; con varios, una lista de DataFrames (None si no encuentra el vídeo).
        """
        if isinstance(X, str):#una sola observación
            return self.__recomendar(X, n)
        return [self.__recomendar(texto, n) for texto in X]

    def save(self, ruta):
        """Guarda en disco (pickle) el estado de la clase: modelo, TF-IDF y vídeos."""
        with open(ruta, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, ruta):
        """Carga de disco el estado guardado con save para poder recomendar sin reentrenar."""
        with open(ruta, "rb") as f:
            self.__dict__.update(pickle.load(f))
        return self

    def __to_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.COLUMNAS].copy()
        return pd.DataFrame(X, columns=self.COLUMNAS)#en un numpy array las columnas tienen que venir en el orden de COLUMNAS

    def __unir_tags(self, tags):
        if isinstance(tags, str):#los tags vienen del CSV separados por "|"
            tags = [tag for tag in tags.replace('"', "").split("|") if tag != ""]#quitamos las etiquetas vacías
        if list(tags) == ["[none]"]:#por si no hubiera ningun tag
            return ""
        return " ".join(tags)

    def __preprocess_data(self, X, entrenar):
        """Junta título, tags, canal y categoría en un solo texto y lo convierte en la matriz TF-IDF."""
        contenido = X["title"] + " " + X["tags"].apply(self.__unir_tags) + " " + X["channel_title"] + " " + X["category"]
        if entrenar:
            return self.preprocess.fit_transform(contenido)
        return self.preprocess.transform(contenido)

    def __recomendar(self, texto, n):
        coincidencias = self.videos[self.videos["title"].str.contains(texto, case=False, regex=False)]
        if coincidencias.empty:
            print("No hemos encontrado ningún vídeo con ese título:", texto)
            return None

        i = coincidencias.index[0]#los vídeos están ordenados por views, así que el primero es el más visto

        distancias, indices = self.modelo.kneighbors(self.X_tfidf[i], n_neighbors=n + 1)

        recomendaciones = self.videos.loc[indices[0], ["title", "channel_title", "category", "pais", "views"]].copy()
        recomendaciones["similitud"] = 1 - distancias[0]
        recomendaciones = recomendaciones.drop(index=i, errors="ignore").head(n)#quitamos el propio vídeo

        return recomendaciones
