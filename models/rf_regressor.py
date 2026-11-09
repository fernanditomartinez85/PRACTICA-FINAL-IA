import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class RFRegressor:
    """Predice los likes: Random Forest con sus parámetros por defecto (R2 en test de 0,949; el optimizado salió peor).

    Se entrena solo con vídeos con los ratings activados, ya que si no los likes valen siempre 0.
    """

    numericas = ["views", "comment_count", "dislikes", "dias_desde_publicacion", "comments_disabled"]
    categoricas = ["pais", "category"]
    columnas = numericas + categoricas

    def __init__(self, random_state=42):
        rf_likes = RandomForestRegressor(random_state=random_state, n_jobs=-1)#n_jobs=-1 solo lo acelera
        #one hot de país y categoría; las numéricas pasan tal cual (los árboles no necesitan logaritmo ni escalado)
        preprocess_rf_likes = ColumnTransformer(transformers=[
            ("encoder", OneHotEncoder(handle_unknown="ignore"), self.categoricas),
            ("numericas", "passthrough", self.numericas)])
        self.pipeline_rf_likes = Pipeline(steps=[("preprocesamiento", preprocess_rf_likes),
                                                 ("modelo", rf_likes)])

    def __preprocess_data(self, X):
        #un vídeo (diccionario o Series) o varios (DataFrame, o numpy array con las columnas en orden)
        if isinstance(X, (dict, pd.Series)):
            X = pd.DataFrame([X])
        elif not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.array(X, dtype=object).reshape(-1, len(self.columnas)), columns=self.columnas)
        return X[self.columnas]

    def fit(self, X, y):
        #ravel: si y llega como columna (numpy array de forma (n, 1)), la deja en una dimensión y evita el aviso de sklearn
        self.pipeline_rf_likes.fit(self.__preprocess_data(X), np.ravel(y))
        return self

    def predict(self, X):
        return self.pipeline_rf_likes.predict(self.__preprocess_data(X))

    def save(self, ruta):
        """Guarda en un pickle todo el estado del modelo (modelo entrenado y transformaciones)."""
        with open(ruta, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, ruta):
        """Recupera el estado guardado con save, para predecir sin reentrenar."""
        with open(ruta, "rb") as f:
            self.__dict__.update(pickle.load(f))
        return self
