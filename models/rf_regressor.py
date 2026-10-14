import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder


class RFRegressor:
    """Random Forest para predecir el número de likes.

    Es el mejor modelo del apartado de predicción de likes (R2 en test de 0,945). El optimizado con RandomizedSearchCV
    salió algo peor, por lo que usamos los parámetros por defecto.
    Se entrena solo con vídeos que tienen los ratings activados (ratings_disabled == False), ya que si no los likes valen 0.
    """

    COLUMNAS = ["views", "comment_count", "dislikes", "dias_desde_publicacion", "comments_disabled", "pais", "category"]
    CATEGORICAS = ["pais", "category"]
    NUMERICAS = ["views", "comment_count", "dislikes", "dias_desde_publicacion", "comments_disabled"]

    def __init__(self, random_state=42):
        self.modelo = RandomForestRegressor(random_state=random_state)

        #En Random Forest no hace falta logaritmo ni escalado en las X: los árboles dividen por umbrales
        self.preprocess = ColumnTransformer(transformers=[
            ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), self.CATEGORICAS),
            ("numericas", "passthrough", self.NUMERICAS)])

    def fit(self, X, y):
        """Entrena el modelo. X: DataFrame o numpy array con las columnas de COLUMNAS. y: likes de cada vídeo."""
        X_procesado = self.__preprocess_data(X, entrenar=True)
        self.modelo.fit(X_procesado, self.__transformar_y(y))
        return self

    def predict(self, X):
        """Devuelve el número de likes de uno o varios vídeos."""
        X_procesado = self.__preprocess_data(X, entrenar=False)
        return self.__deshacer_y(self.modelo.predict(X_procesado))

    def save(self, ruta):
        """Guarda en disco (pickle) el estado de la clase: modelo y preprocess."""
        with open(ruta, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, ruta):
        """Carga de disco el estado guardado con save para poder predecir sin reentrenar."""
        with open(ruta, "rb") as f:
            self.__dict__.update(pickle.load(f))
        return self

    def __to_dataframe(self, X):
        """Convierte la entrada a DataFrame, tanto si es una sola observación como un conjunto."""
        if isinstance(X, pd.DataFrame):
            return X[self.COLUMNAS]
        if isinstance(X, pd.Series):#una sola observación
            return X.to_frame().T[self.COLUMNAS]
        if isinstance(X, dict):#una sola observación
            return pd.DataFrame([X])[self.COLUMNAS]
        X = np.array(X, dtype=object)
        if X.ndim == 1:#una sola observación
            X = X.reshape(1, -1)
        return pd.DataFrame(X, columns=self.COLUMNAS)#en un numpy array las columnas tienen que venir en el orden de COLUMNAS

    def __preprocess_data(self, X, entrenar):
        """Aplica el one-hot a pais y category. Al entrenar ajusta el preprocess; al predecir solo lo aplica."""
        X = self.__to_dataframe(X)
        X = X.astype({col: float for col in self.NUMERICAS})#comments_disabled pasa de True/False a 1/0
        if entrenar:
            return self.preprocess.fit_transform(X)
        return self.preprocess.transform(X)

    def __transformar_y(self, y):
        """Los likes están muy sesgados, así que entrenamos con log1p(likes), igual que con TransformedTargetRegressor."""
        return np.log1p(np.ravel(y).astype(float))

    def __deshacer_y(self, y_log):
        """Deshace el logaritmo para que las predicciones vuelvan a estar en número de likes."""
        return np.expm1(y_log)
