import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder


class RFRegressor:

    def __init__(self):

        encoder = OneHotEncoder(handle_unknown="ignore")
        self.rf_likes = RandomForestRegressor(random_state=42, n_jobs=-1)
        self.preprocess_rf_likes = ColumnTransformer(transformers=[
            ("encoder", encoder, ["pais", "category"]),
            ("numericas", "passthrough",
             ["views", "comment_count", "dislikes", "dias_desde_publicacion", "comments_disabled"])])

    def __preprocess_data(self, df, y=None):
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(np.array(df, dtype=object).reshape(-1, 7),#valores en filas de 7 columnas
                              columns=["views", "comment_count", "dislikes", "dias_desde_publicacion",
                                       "comments_disabled", "pais", "category"])

        X_likes = df[["views", "comment_count", "dislikes", "dias_desde_publicacion",
                      "comments_disabled", "pais", "category"]]
        if y is None:#al predecir no hay y
            X_likes = self.preprocess_rf_likes.transform(X_likes)
            return X_likes

        X_likes = self.preprocess_rf_likes.fit_transform(X_likes)
        return X_likes, y

    def fit(self, df, y):
        X_likes, y_likes = self.__preprocess_data(df, y)
        self.rf_likes.fit(X_likes, y_likes)
        return self

    def predict(self, df):
        X_likes = self.__preprocess_data(df)
        y_pred_likes = self.rf_likes.predict(X_likes)
        return y_pred_likes

    def save(self, ruta):
        with open(ruta, "wb") as f:
            pickle.dump(self, f)

    def load(self, ruta):
        with open(ruta, "rb") as f:
            modelo_entrenado = pickle.load(f)
        return modelo_entrenado
