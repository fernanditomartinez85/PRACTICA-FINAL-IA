import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class RFClassifier:

    def __init__(self):
        
        self.rf = RandomForestClassifier(n_estimators=150, criterion="gini", max_depth=20, min_samples_split=2,
                                         min_samples_leaf=1, max_features="log2", class_weight="balanced_subsample",
                                         bootstrap=True, max_samples=0.8, random_state=42, n_jobs=-1)

    def __preprocess_data(self, df, y=None):
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(np.array(df, dtype=object).reshape(-1, 6),#valores en filas de 6 columnas
                              columns=["views", "likes", "dislikes", "comment_count", "dias_desde_publicacion", "pais"])
            
        X = df[["views", "likes", "dislikes", "comment_count", "dias_desde_publicacion", "pais"]]
        if y is None:#al predecir no hay y
            X = self.preprocess_rf.transform(X)
            return X
        
        encoder = OneHotEncoder(handle_unknown="ignore")
        self.preprocess_rf = ColumnTransformer(transformers=[
            ("encoder", encoder, ["pais"]),
            ("numericas", "passthrough",
             ["views", "likes", "dislikes", "comment_count", "dias_desde_publicacion"])])
        X = self.preprocess_rf.fit_transform(X)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(y)
        y = self.label_encoder.transform(y)
        return X, y

    def fit(self, df, y):
        X, y = self.__preprocess_data(df, y)
        self.rf.fit(X, y)
        return self

    def predict(self, df):
        X = self.__preprocess_data(df)
        y_pred = self.rf.predict(X)
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
