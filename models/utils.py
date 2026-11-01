import pickle

import numpy as np
import pandas as pd


def a_dataframe(X, columnas):
    """Pasa a DataFrame un vídeo (diccionario o Series) o varios (DataFrame, o numpy array con las columnas en orden)."""
    if isinstance(X, (dict, pd.Series)):
        X = pd.DataFrame([X])
    elif not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(np.array(X, dtype=object).reshape(-1, len(columnas)), columns=columnas)
    return X[columnas]


def guardar(modelo, ruta):
    """Guarda en un pickle todo el estado del modelo (modelo entrenado y transformaciones)."""
    with open(ruta, "wb") as f:
        pickle.dump(modelo.__dict__, f)


def cargar(modelo, ruta):
    """Recupera el estado guardado con guardar, para predecir sin reentrenar."""
    with open(ruta, "rb") as f:
        modelo.__dict__.update(pickle.load(f))
    return modelo
