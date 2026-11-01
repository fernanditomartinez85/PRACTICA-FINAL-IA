import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class RFClassifier:
    """Predice la categoría del vídeo: Random Forest optimizado con RandomizedSearchCV (F1 macro en test de 0,32)."""

    NUMERICAS = ["views", "likes", "dislikes", "comment_count", "dias_desde_publicacion"]
    CATEGORICAS = ["pais"]
    COLUMNAS = NUMERICAS + CATEGORICAS

    def __init__(self, n_estimators=150, criterion="gini", max_depth=20, min_samples_split=2, min_samples_leaf=1,
                 max_features="log2", class_weight="balanced_subsample", max_samples=0.8, random_state=42):
        #por defecto, los mejores parámetros del RandomizedSearchCV del notebook; n_jobs=-1 solo lo acelera
        self.modelo = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                             min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                             max_features=max_features, class_weight=class_weight, bootstrap=True,
                                             max_samples=max_samples, random_state=random_state, n_jobs=-1)
        #one hot del país; las numéricas pasan tal cual (los árboles no necesitan logaritmo ni escalado)
        self.preprocess = ColumnTransformer([("encoder", OneHotEncoder(handle_unknown="ignore"), self.CATEGORICAS),
                                             ("numericas", "passthrough", self.NUMERICAS)])
        self.label_encoder = LabelEncoder()#categorías de texto a números y al revés

    def fit(self, X, y):
        self.modelo.fit(self.__preprocess_data(X, entrenar=True), self.label_encoder.fit_transform(np.ravel(y)))
        return self

    def predict(self, X):
        return self.label_encoder.inverse_transform(self.modelo.predict(self.__preprocess_data(X, entrenar=False)))

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
            X = pd.DataFrame(np.array(X, dtype=object).reshape(-1, len(self.COLUMNAS)), columns=self.COLUMNAS)
        return X[self.COLUMNAS]

    def __preprocess_data(self, X, entrenar):
        X = self.__a_dataframe(X).astype({col: float for col in self.NUMERICAS})
        return self.preprocess.fit_transform(X) if entrenar else self.preprocess.transform(X)
