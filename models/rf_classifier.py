import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class RFClassifier:
    """Random Forest optimizado para predecir la categoría del vídeo.

    Es el mejor modelo del apartado de predicción de categoría (RandomizedSearchCV, F1 macro en test de 0,32).
    """

    COLUMNAS = ["views", "likes", "dislikes", "comment_count", "dias_desde_publicacion", "pais"]
    CATEGORICAS = ["pais"]
    NUMERICAS = ["views", "likes", "dislikes", "comment_count", "dias_desde_publicacion"]

    def __init__(self, n_estimators=150, criterion="gini", max_depth=20, min_samples_split=2, min_samples_leaf=1,
                 max_features="log2", class_weight="balanced_subsample", max_samples=0.8, random_state=42):
        #Por defecto, los mejores parámetros que encontró RandomizedSearchCV en el notebook
        self.modelo = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                             min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                             max_features=max_features, class_weight=class_weight, bootstrap=True,
                                             max_samples=max_samples, random_state=random_state)

        #En Random Forest no hace falta logaritmo ni escalado: los árboles dividen por umbrales y no les afecta la magnitud
        self.preprocess = ColumnTransformer(transformers=[
            ("encoder", OneHotEncoder(handle_unknown="ignore"), self.CATEGORICAS),#ignore: un país o categoría que no vio al entrenar se deja a 0 en vez de dar error
            ("numericas", "passthrough", self.NUMERICAS)])

        self.label_encoder = LabelEncoder()#pasa las categorías (texto) a números y al revés

    def fit(self, X, y):
        """Entrena el modelo. X: DataFrame o numpy array con las columnas de COLUMNAS. y: categoría de cada vídeo."""
        X_procesado = self.__preprocess_data(X, entrenar=True)
        y_codificada = self.label_encoder.fit_transform(np.ravel(y))
        self.modelo.fit(X_procesado, y_codificada)
        return self

    def predict(self, X):
        """Devuelve la categoría (en texto) de uno o varios vídeos."""
        X_procesado = self.__preprocess_data(X, entrenar=False)
        return self.label_encoder.inverse_transform(self.modelo.predict(X_procesado))

    def save(self, ruta):
        """Guarda en disco (pickle) el estado de la clase: modelo, preprocess y label_encoder."""
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
        """Aplica el one-hot a pais. Al entrenar ajusta el preprocess; al predecir solo lo aplica."""
        X = self.__to_dataframe(X)
        X = X.astype({col: float for col in self.NUMERICAS})
        if entrenar:
            return self.preprocess.fit_transform(X)
        return self.preprocess.transform(X)
