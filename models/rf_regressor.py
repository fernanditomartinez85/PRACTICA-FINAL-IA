import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

from models.utils import a_dataframe, cargar, guardar


class RFRegressor:
    """Predice los likes: Random Forest con sus parámetros por defecto (R2 en test de 0,949; el optimizado salió peor).

    Se entrena solo con vídeos con los ratings activados, ya que si no los likes valen siempre 0.
    """

    NUMERICAS = ["views", "comment_count", "dislikes", "dias_desde_publicacion", "comments_disabled"]
    CATEGORICAS = ["pais", "category"]
    COLUMNAS = NUMERICAS + CATEGORICAS

    def __init__(self, random_state=42):
        self.modelo = RandomForestRegressor(random_state=random_state, n_jobs=-1)#n_jobs=-1 solo lo acelera
        #one hot de país y categoría; las numéricas pasan tal cual (los árboles no necesitan logaritmo ni escalado)
        self.preprocess = ColumnTransformer([("encoder", OneHotEncoder(handle_unknown="ignore"), self.CATEGORICAS),
                                             ("numericas", "passthrough", self.NUMERICAS)])

    def fit(self, X, y):
        self.modelo.fit(self.__preprocess_data(X, entrenar=True), np.ravel(y))
        return self

    def predict(self, X):
        return self.modelo.predict(self.__preprocess_data(X, entrenar=False))

    def save(self, ruta):
        guardar(self, ruta)

    def load(self, ruta):
        return cargar(self, ruta)

    def __preprocess_data(self, X, entrenar):
        #comments_disabled pasa de True/False a 1/0
        X = a_dataframe(X, self.COLUMNAS).astype({col: float for col in self.NUMERICAS})
        return self.preprocess.fit_transform(X) if entrenar else self.preprocess.transform(X)
