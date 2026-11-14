import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, normalize

class KNNRecommender:

    def __init__(self):
        
        self.mlb_titulo = MultiLabelBinarizer(sparse_output=True)
        self.mlb_tags = MultiLabelBinarizer(sparse_output=True)
        self.mlb_canal = MultiLabelBinarizer(sparse_output=True)

    def __preprocess_data(self, df):
        palabras_titulo = df["title"].str.lower().str.split()
        df["tags"] = df["tags"].str.split("|")
        lista_tags = df["tags"].apply(self.__lista_tags)
        lista_canal = []#cada nombre de canal una etiqueta con el nombre entero, por eso no usamos split.
        for canal in df["channel_title"]:
            lista_canal.append([canal])
        return (normalize(self.mlb_titulo.fit_transform(palabras_titulo)),
                normalize(self.mlb_tags.fit_transform(lista_tags)),
                normalize(self.mlb_canal.fit_transform(lista_canal)))

    def __lista_tags(self, tags):
        if tags == ["[none]"]:
            return []
        return [tag.lower() for tag in tags]

    def fit(self, df, y=None):#y=None: train_models.py llama a fit con dos argumentos, aunque aquí no hay y
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(np.array(df, dtype=object).reshape(-1, 6),#valores en filas de 6 columnas
                             columns=["title", "tags", "channel_title", "category", "pais", "views"])
        df = df.sort_values(by="views", ascending=False)
        df = df.drop_duplicates(subset=["title", "channel_title"], keep="first")
        self.videos = df.reset_index(drop=True)
        self.X_titulo, self.X_tags, self.X_canal = self.__preprocess_data(self.videos)#no hay modelo que entrenar
        return self

    def __distancia_videos(self, i):
        #distancia del coseno = 1 - similitud del coseno (el producto de las filas normalizadas)
        distancia_titulo = 1 - (self.X_titulo @ self.X_titulo[i].T).toarray().ravel()
        distancia_tags = 1 - (self.X_tags @ self.X_tags[i].T).toarray().ravel()
        distancia_canal = 1 - (self.X_canal @ self.X_canal[i].T).toarray().ravel()
        return distancia_titulo + distancia_tags + distancia_canal

    def __recomendar_variado(self, titulo=None, canal=None, categoria=None, n=10, umbral=0.5):
        if titulo is None and canal is None and categoria is None:
            print("Indica al menos el título, el canal o la categoría")
            return None

        coincidencias = self.videos
        if titulo is not None:
            coincidencias = coincidencias[coincidencias["title"].str.contains(titulo, case=False, regex=False)]
        if canal is not None:
            coincidencias = coincidencias[coincidencias["channel_title"] == canal]
        if categoria is not None:
            coincidencias = coincidencias[coincidencias["category"] == categoria]
        if coincidencias.empty:
            print("No hemos encontrado ningún vídeo con esos datos")
            return None

        i = coincidencias.index[0]#self.videos está ordenado por views, así que el primero es el más visto
        print("Vídeo elegido:", self.videos.loc[i, "title"], "|", self.videos.loc[i, "channel_title"], "|",
              self.videos.loc[i, "category"])

        #candidatos: los 500 vídeos más cercanos, ordenados de más a menos parecido
        #Vamos a ir descartando vídeos repetidos, y si pidiéramos solo 10 nos podríamos quedar con 3.
        candidatos = self.videos[["title", "channel_title", "category", "pais", "views"]].copy()
        candidatos["distancia"] = self.__distancia_videos(i)
        candidatos = candidatos.sort_values(by="distancia")
        candidatos = candidatos.drop(index=i).head(500).drop(columns="distancia")

        elegidos = []
        for video in candidatos.index:
            video_seleccionado = True
            for elegido in elegidos:
                #si se parece más de 0,5 (umbral), no se selecciona
                if (self.X_titulo[video] @ self.X_titulo[elegido].T).toarray()[0, 0] > umbral:
                    video_seleccionado = False
            if video_seleccionado:
                elegidos.append(video)
            if len(elegidos) == n:
                break

        return candidatos.loc[elegidos]

    def predict(self, df, n=10):
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(np.array(df, dtype=object).reshape(-1, 3),#valores en filas de 3 columnas
                              columns=["title", "channel_title", "category"])
        recomendaciones = []
        df = df.fillna("")
        df = df.replace("", None)#lo que no se indica (vacío) pasa a None, que es lo que entiende __recomendar_variado
        for titulo, canal, categoria in zip(df["title"], df["channel_title"], df["category"]):
            recomendaciones.append(self.__recomendar_variado(titulo, canal, categoria, n))
        return recomendaciones

    def save(self, ruta):
        with open(ruta, "wb") as f:
            pickle.dump(self, f)

    def load(self, ruta):
        with open(ruta, "rb") as f:
            modelo_entrenado = pickle.load(f)
        return modelo_entrenado
