import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class RFClassifier:

    def __init__(self, n_estimators=150, criterion="gini", max_depth=20, min_samples_split=2, min_samples_leaf=1,
                 max_features="log2", class_weight="balanced_subsample", max_samples=0.8, random_state=42):
        
        rf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                    min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                    max_features=max_features, class_weight=class_weight, bootstrap=True,
                                    max_samples=max_samples, random_state=random_state, n_jobs=-1)
        #one hot del país; las numéricas pasan tal cual (los árboles no necesitan logaritmo ni escalado)
        preprocess_rf = ColumnTransformer(transformers=[
            ("encoder", OneHotEncoder(handle_unknown="ignore"), ["pais"]),
            ("numericas", "passthrough",
             ["views", "likes", "dislikes", "comment_count", "dias_desde_publicacion"])])
        self.pipeline_rf = Pipeline(steps=[("preprocesamiento", preprocess_rf),
                                           ("modelo", rf)])
        self.label_encoder = LabelEncoder()#categorías de texto a números y al revés

    def __preprocess_data(self, X):
        columnas = ["views", "likes", "dislikes", "comment_count", "dias_desde_publicacion", "pais"]
        #un vídeo (diccionario o Series) o varios (DataFrame, o numpy array con las columnas en orden)
        if isinstance(X, (dict, pd.Series)):
            X = pd.DataFrame([X])
        elif not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.array(X, dtype=object).reshape(-1, len(columnas)), columns=columnas)
        return X[columnas]

    def fit(self, X, y):
        #ravel: si y llega como columna (numpy array de forma (n, 1)), la deja en una dimensión y evita el aviso de sklearn
        self.pipeline_rf.fit(self.__preprocess_data(X), self.label_encoder.fit_transform(np.ravel(y)))
        return self

    def predict(self, X):
        return self.label_encoder.inverse_transform(self.pipeline_rf.predict(self.__preprocess_data(X)))

    def save(self, ruta):
        """Guarda en un pickle todo el estado del modelo (modelo entrenado y transformaciones)."""
        with open(ruta, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, ruta):
        """Recupera el estado guardado con save, para predecir sin reentrenar."""
        with open(ruta, "rb") as f:
            self.__dict__.update(pickle.load(f))
        return self
