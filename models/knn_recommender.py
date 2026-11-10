import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, normalize


class KNNRecommender:
    """Recomienda vídeos parecidos por el título, los tags y el canal (one hot y suma de distancias del coseno).

    Para no recomendar muchas versiones del mismo vídeo, se salta los que tienen un título demasiado parecido al de una
    recomendación ya elegida (recomendar_variado del notebook).
    """

    columnas = ["title", "tags", "channel_title", "category", "pais", "views"]
    mostrar = ["title", "channel_title", "category", "pais", "views"]
    busqueda = ["title", "channel_title", "category"]#datos con los que se elige el vídeo de partida
    n_candidatos = 500#vídeos más cercanos entre los que se eligen las recomendaciones
    umbral = 0.5#similitud del título a partir de la cual dos vídeos cuentan como versiones del mismo

    def __init__(self):
        #sparse_output=True: solo guarda los 1, ya que cada matriz tiene cientos de miles de columnas
        self.mlb_titulo = MultiLabelBinarizer(sparse_output=True)
        self.mlb_tags = MultiLabelBinarizer(sparse_output=True)
        self.mlb_canal = MultiLabelBinarizer(sparse_output=True)
        self.videos = self.X_titulo = self.X_tags = self.X_canal = None#se calculan en fit

    def __preprocess_data(self, X):
        """Pasa las palabras del título, los tags y el canal a matrices de 0 y 1 con cada fila normalizada."""
        palabras_titulo = X["title"].str.lower().str.split()
        lista_tags = X["tags"].apply(self.__lista_tags)
        #MultiLabelBinarizer espera una lista por vídeo: con el texto suelto crearía una columna por letra
        lista_canal = []
        for canal in X["channel_title"]:
            lista_canal.append([canal])
        #con las filas normalizadas, la similitud del coseno entre dos vídeos es multiplicar sus filas
        return (normalize(self.mlb_titulo.fit_transform(palabras_titulo)),
                normalize(self.mlb_tags.fit_transform(lista_tags)),
                normalize(self.mlb_canal.fit_transform(lista_canal)))

    def __lista_tags(self, tags):
        if isinstance(tags, str):#del csv vienen separados por "|"
            tags = [tag for tag in tags.replace('"', "").split("|") if tag != ""]
        return [] if list(tags) == ["[none]"] else [tag.lower() for tag in tags]

    def fit(self, X, y=None):
        #un vídeo (diccionario o Series) o varios (DataFrame, o numpy array con las columnas en orden)
        if isinstance(X, (dict, pd.Series)):
            X = pd.DataFrame([X])
        elif not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.array(X, dtype=object).reshape(-1, len(self.columnas)), columns=self.columnas)
        #un registro por vídeo (el de más views), con el índice igual a su fila en las matrices
        videos = X[self.columnas].sort_values(by="views", ascending=False)
        self.videos = videos.drop_duplicates(subset=["title", "channel_title"]).reset_index(drop=True)
        self.X_titulo, self.X_tags, self.X_canal = self.__preprocess_data(self.videos)
        return self

    def __distancia_videos(self, i):
        """Suma de las distancias del coseno del vídeo i a todos los vídeos: de 0 (iguales) a 3 (nada en común)."""
        return sum(1 - (X @ X[i].T).toarray().ravel() for X in (self.X_titulo, self.X_tags, self.X_canal))

    def __recomendar(self, datos, n):
        #solo filtramos por lo que venga indicado (en un csv, lo que no se indica llega vacío)
        indicados = {col: datos.get(col) for col in self.busqueda if datos.get(col) is not None
                     and not pd.isna(datos.get(col)) and datos.get(col) != ""}
        if not indicados:
            print("Indica al menos el título, el canal o la categoría")
            return None
        coincidencias = self.videos
        for col, valor in indicados.items():
            if col == "title":#basta con que el título contenga el texto
                coincidencias = coincidencias[coincidencias[col].str.contains(valor, case=False, regex=False)]
            else:
                coincidencias = coincidencias[coincidencias[col] == valor]
        if coincidencias.empty:
            print("No hemos encontrado ningún vídeo con esos datos:", indicados)
            return None
        i = coincidencias.index[0]#el más visto, porque los vídeos están ordenados por views

        #candidatos del más cercano al más lejano, sin el propio vídeo
        candidatos = pd.Series(self.__distancia_videos(i)).sort_values().drop(i).index[:self.n_candidatos]
        #elegimos de uno en uno y saltamos los que tienen un título demasiado parecido al de uno ya elegido
        elegidos = []
        for video in candidatos:
            video_seleccionado = True
            for e in elegidos:
                #si se parece más que umbral a alguno de los ya elegidos, no se selecciona
                if (self.X_titulo[video] @ self.X_titulo[e].T)[0, 0] > self.umbral:
                    video_seleccionado = False
            if video_seleccionado:
                elegidos.append(video)
                if len(elegidos) == n:
                    break
        return self.videos.loc[elegidos, self.mostrar]

    def predict(self, X, n=10):
        """Recibe el título, el canal o la categoría (uno, dos o los tres) de uno o varios vídeos y recomienda n vídeos.

        Con un vídeo (diccionario o Series) devuelve un DataFrame; con varios, una lista (None si no encuentra el vídeo).
        """
        if isinstance(X, (dict, pd.Series)):
            return self.__recomendar(X, n)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.busqueda)
        return [self.__recomendar(fila, n) for _, fila in X.iterrows()]

    def save(self, ruta):
        with open(ruta, "wb") as f:
            pickle.dump(self, f)

    def load(self, ruta):
        """Recupera el estado guardado con save, para predecir sin reentrenar."""
        with open(ruta, "rb") as f:
            modelo_entrenado = pickle.load(f)
        return modelo_entrenado
