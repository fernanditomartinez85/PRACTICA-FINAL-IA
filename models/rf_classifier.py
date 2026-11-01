import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from models.utils import a_dataframe, cargar, guardar


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
        guardar(self, ruta)

    def load(self, ruta):
        return cargar(self, ruta)

    def __preprocess_data(self, X, entrenar):
        X = a_dataframe(X, self.COLUMNAS).astype({col: float for col in self.NUMERICAS})
        return self.preprocess.fit_transform(X) if entrenar else self.preprocess.transform(X)
