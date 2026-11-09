import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class RFClassifier:

    def __init__(self):
        
        rf = RandomForestClassifier(n_estimators=150, criterion="gini", max_depth=20, min_samples_split=2,
                                    min_samples_leaf=1, max_features="log2", class_weight="balanced_subsample",
                                    bootstrap=True, max_samples=0.8, random_state=42, n_jobs=-1)
        
        preprocess_rf = ColumnTransformer(transformers=[
            ("encoder", OneHotEncoder(handle_unknown="ignore"), ["pais"]),
            ("numericas", "passthrough",
             ["views", "likes", "dislikes", "comment_count", "dias_desde_publicacion"])])
        
        self.pipeline_rf = Pipeline(steps=[("preprocesamiento", preprocess_rf),
                                           ("modelo", rf)])
        
        self.label_encoder = LabelEncoder()

    def __preprocess_data(self, X, y=None):
        columnas = ["views", "likes", "dislikes", "comment_count", "dias_desde_publicacion", "pais"]
        #un vídeo (diccionario o Series) o varios (DataFrame, o numpy array con las columnas en orden)
        if isinstance(X, (dict, pd.Series)):
            X = pd.DataFrame([X])
        elif not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.array(X, dtype=object).reshape(-1, len(columnas)), columns=columnas)
        X = X[columnas]
        if y is None:#al predecir no hay y
            return X
        #ravel: si y llega como columna (numpy array de forma (n, 1)), la deja en una dimensión y evita el aviso de sklearn
        y = np.ravel(y)
        self.label_encoder.fit(y)
        y = self.label_encoder.transform(y)
        return X, y

    def fit(self, X, y):
        X, y = self.__preprocess_data(X, y)
        self.pipeline_rf.fit(X, y)
        return self

    def predict(self, X):
        y_pred = self.pipeline_rf.predict(self.__preprocess_data(X))
        return self.label_encoder.inverse_transform(y_pred)

    def save(self, ruta):
        """Guarda en un pickle todo el estado del modelo (modelo entrenado y transformaciones)."""
        with open(ruta, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, ruta):
        """Recupera el estado guardado con save, para predecir sin reentrenar."""
        with open(ruta, "rb") as f:
            self.__dict__.update(pickle.load(f))
        return self
