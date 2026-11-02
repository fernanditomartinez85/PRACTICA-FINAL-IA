import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder


class RFRegressor:
    """Predice los likes: Random Forest con sus parámetros por defecto (R2 en test de 0,949; el optimizado salió peor).

    Se entrena solo con vídeos con los ratings activados, ya que si no los likes valen siempre 0.
    """

    numericas = ["views", "comment_count", "dislikes", "dias_desde_publicacion", "comments_disabled"]
    categoricas = ["pais", "category"]
    columnas = numericas + categoricas

    def __init__(self, random_state=42):
        self.modelo = RandomForestRegressor(random_state=random_state, n_jobs=-1)#n_jobs=-1 solo lo acelera
        #one hot de país y categoría; las numéricas pasan tal cual (los árboles no necesitan logaritmo ni escalado)
        self.preprocess = ColumnTransformer([("encoder", OneHotEncoder(handle_unknown="ignore"), self.categoricas),
                                             ("numericas", "passthrough", self.numericas)])

    def fit(self, X, y):
        self.modelo.fit(self.__preprocess_data(X, entrenar=True), np.ravel(y))
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

    def __a_dataframe(self, X):
        """Pasa a DataFrame un vídeo (diccionario o Series) o varios (DataFrame, o numpy array con las columnas en orden)."""
        if isinstance(X, (dict, pd.Series)):
            X = pd.DataFrame([X])
        elif not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.array(X, dtype=object).reshape(-1, len(self.columnas)), columns=self.columnas)
        return X[self.columnas]

    def __preprocess_data(self, X, entrenar):
        #comments_disabled pasa de True/False a 1/0
        X = self.__a_dataframe(X).astype({col: float for col in self.numericas})
        return self.preprocess.fit_transform(X) if entrenar else self.preprocess.transform(X)
