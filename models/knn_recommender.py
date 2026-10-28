import pickle

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, normalize


class KNNRecommender:
    """Recomendador basado en contenido: one hot de las palabras del título, los tags y el canal, y los vecinos más
    cercanos con la suma de las distancias del coseno de las tres variables, saltando los que tienen un título demasiado
    parecido al de una recomendación ya elegida (recomendar_variado del notebook).

    Es el modelo del apartado del recomendador.
    """

    COLUMNAS = ["title", "tags", "channel_title", "category", "pais", "views"]
    CANDIDATOS = 500#vídeos más cercanos de entre los que se eligen las recomendaciones
    UMBRAL = 0.5#similitud del título a partir de la cual dos vídeos se consideran versiones del mismo

    def __init__(self):
        #un one hot para múltiples etiquetas por variable; sparse_output=True: solo guarda los 1, ya que cada matriz
        #tiene cientos de miles de columnas
        self.mlb_titulo = MultiLabelBinarizer(sparse_output=True)
        self.mlb_tags = MultiLabelBinarizer(sparse_output=True)
        self.mlb_canal = MultiLabelBinarizer(sparse_output=True)
        self.videos = None#se guardan en fit: son los vídeos entre los que se recomienda
        self.X_titulo = None
        self.X_tags = None
        self.X_canal = None

    def fit(self, X, y=None):
        """Entrena el recomendador. X: DataFrame o numpy array con las columnas de COLUMNAS (no necesita y)."""
        videos = self.__to_dataframe(X)
        videos = videos.sort_values(by="views", ascending=False)
        videos = videos.drop_duplicates(subset=["title", "channel_title"], keep="first")#un único registro por vídeo para no recomendar el mismo vídeo varias veces
        self.videos = videos.reset_index(drop=True)#cada fila coincide con su posición en las matrices

        self.X_titulo, self.X_tags, self.X_canal = self.__preprocess_data(self.videos)
        return self

    def predict(self, X, n=10):
        """Recibe uno o varios vídeos con su título, su canal o su categoría (title, channel_title y category; basta con
        uno de los tres) y devuelve los n vídeos más parecidos al vídeo más visto que cumple lo indicado.

        Con un solo vídeo (diccionario o Series) devuelve un DataFrame; con varios (DataFrame, o numpy array con esas
        tres columnas en ese orden), una lista de DataFrames (None si no encuentra el vídeo).
        """
        if isinstance(X, (dict, pd.Series)):#una sola observación
            return self.__recomendar(X.get("title"), X.get("channel_title"), X.get("category"), n)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=["title", "channel_title", "category"])
        return [self.__recomendar(fila.get("title"), fila.get("channel_title"), fila.get("category"), n)
                for _, fila in X.iterrows()]

    def save(self, ruta):
        """Guarda en disco (pickle) el estado de la clase: one hot de cada variable, matrices y vídeos."""
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

    def __lista_tags(self, tags):
        if isinstance(tags, str):#los tags vienen del CSV separados por "|"
            tags = [tag for tag in tags.replace('"', "").split("|") if tag != ""]#quitamos las etiquetas vacías
        if list(tags) == ["[none]"]:#por si no hubiera ningun tag
            return []
        return [tag.lower() for tag in tags]

    def __preprocess_data(self, X):
        """Convierte las palabras del título, los tags y el canal en matrices de 0 y 1, con cada fila normalizada."""
        palabras_titulo = X["title"].str.lower().str.split()
        lista_tags = X["tags"].apply(self.__lista_tags)
        #el canal es un único texto, pero MultiLabelBinarizer espera una lista por vídeo (como el título y los tags).
        #Si le pasamos el texto, lo recorre letra a letra y crea una columna por letra; metiéndolo en una lista de un
        #solo elemento, cada canal es una única etiqueta
        lista_canal = []
        for canal in X["channel_title"]:
            lista_canal.append([canal])

        #normalizamos cada fila para que la similitud del coseno entre dos vídeos sea multiplicar sus filas
        X_titulo = normalize(self.mlb_titulo.fit_transform(palabras_titulo))
        X_tags = normalize(self.mlb_tags.fit_transform(lista_tags))
        X_canal = normalize(self.mlb_canal.fit_transform(lista_canal))
        return X_titulo, X_tags, X_canal

    def __distancia_videos(self, i):
        """Suma de las distancias del coseno (1 - similitud) del vídeo i a todos los vídeos: de 0 (iguales) a 3."""
        distancia_titulo = 1 - (self.X_titulo @ self.X_titulo[i].T).toarray().ravel()
        distancia_tags = 1 - (self.X_tags @ self.X_tags[i].T).toarray().ravel()
        distancia_canal = 1 - (self.X_canal @ self.X_canal[i].T).toarray().ravel()
        return distancia_titulo + distancia_tags + distancia_canal

    def __vacio(self, valor):
        #un dato que no nos indican: no viene, viene vacío ("" en un CSV) o es nulo
        return valor is None or (isinstance(valor, float) and pd.isna(valor)) or valor == ""

    def __recomendar(self, titulo, canal, categoria, n):
        if self.__vacio(titulo) and self.__vacio(canal) and self.__vacio(categoria):
            print("Indica al menos el título, el canal o la categoría")
            return None

        #filtramos solo por lo que nos indiquen: el título que contiene el texto, el canal y la categoría
        coincidencias = self.videos
        if not self.__vacio(titulo):
            coincidencias = coincidencias[coincidencias["title"].str.contains(titulo, case=False, regex=False)]
        if not self.__vacio(canal):
            coincidencias = coincidencias[coincidencias["channel_title"] == canal]
        if not self.__vacio(categoria):
            coincidencias = coincidencias[coincidencias["category"] == categoria]
        if coincidencias.empty:
            print("No hemos encontrado ningún vídeo con esos datos:", titulo, "|", canal, "|", categoria)
            return None

        i = coincidencias.index[0]#los vídeos están ordenados por views, así que el primero es el más visto

        #trabajamos sobre una copia de las columnas que mostramos para no añadir la distancia a self.videos
        recomendaciones = self.videos[["title", "channel_title", "category", "pais", "views"]].copy()
        recomendaciones["distancia"] = self.__distancia_videos(i)
        recomendaciones = recomendaciones.sort_values(by="distancia")
        #quitamos el propio vídeo, que es el más cercano (distancia 0), nos quedamos con los CANDIDATOS más cercanos y
        #quitamos la distancia, que solo nos sirve para ordenar
        candidatos = recomendaciones.drop(index=i).head(self.CANDIDATOS).drop(columns="distancia")

        #para que no recomiende muchas versiones del mismo vídeo, elegimos de uno en uno, del más cercano al más lejano,
        #y nos saltamos los que tienen un título demasiado parecido al de uno ya elegido
        elegidos = []
        for video in candidatos.index:
            #similitud del título con cada vídeo ya elegido (las filas de X_titulo están normalizadas)
            similitudes = [(self.X_titulo[video] @ self.X_titulo[elegido].T).toarray()[0, 0] for elegido in elegidos]
            if all(similitud <= self.UMBRAL for similitud in similitudes):
                elegidos.append(video)
            if len(elegidos) == n:
                break

        return candidatos.loc[elegidos]
