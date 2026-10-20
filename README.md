# 📺 YouTube Trending Videos

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) ![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square) ![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat-square) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

> Práctica final de IA: clasificación, regresión, clustering y recomendación sobre los vídeos en tendencia de YouTube.

## 🎯 Objetivo

A partir de los datos de los vídeos en tendencias diarias de YouTube de 10 países, se resuelven varios problemas de negocio con técnicas de machine learning:

- **Clasificación:** predecir la categoría del vídeo. Como reto opcional, predecir si el creador desactivará los comentarios o los ratings (solo en el notebook).
- **Regresión:** predecir el número de likes.
- **Clusterización:** encontrar grupos en los vídeos con distintas técnicas, compararlos con las categorías y usar técnicas de reducción de dimensionalidad.
- **Recomendación:** dado un vídeo, recomendar vídeos parecidos.

## 📊 Datos

Vídeos en tendencias diarias de YouTube de 10 países (CA, DE, FR, GB, IN, JP, KR, MX, RU, US). Por cada país hay un CSV con los vídeos (título, canal, fecha de publicación, tags, views, likes, dislikes, comentarios...) y un JSON con el nombre de cada categoría.

## 🗂️ Estructura del proyecto

```
├── XXvideos.csv / XX_category_id.json   Datos en crudo de cada país (CA, DE, FR, GB, IN, JP, KR, MX, RU, US)
├── exploratory_analysis.ipynb           Análisis exploratorio, comparación de modelos y búsqueda de hiperparámetros
├── preprocess_data.py                   Limpieza y transformaciones comunes a todos los modelos
├── train_models.py                      Entrena los modelos y los guarda en disco
├── inference_model.py                   Carga un modelo entrenado y hace inferencia sobre nuevos datos
├── train.conf                           Configuración de rutas para train_models.py
└── models/                              Paquete con los modelos industrializados
    ├── rf_classifier.py                 RFClassifier: categoría del vídeo
    ├── rf_regressor.py                  RFRegressor: número de likes
    ├── kmeans_clustering.py             KMeansClustering: grupos de vídeos
    └── knn_recommender.py               KNNRecommender: recomendador de vídeos
```

## ⚙️ Requisitos

Python 3 con pandas, numpy, scikit-learn, matplotlib y seaborn (por ejemplo, el entorno `base` de Anaconda).

## ▶️ Ejecución

Todos los comandos se lanzan desde la carpeta del proyecto.

**1. Preprocesamiento**

```
python preprocess_data.py <carpeta_datos_crudos> <fichero_salida.csv>
python preprocess_data.py . datos_limpios.csv
```

Recibe la carpeta con los datos en crudo y la ruta del CSV de salida. Realiza los pasos comunes a todos los modelos:

- Cruza el CSV de cada país con su JSON para sustituir category_id por el nombre de la categoría.
- Junta los datos de todos los países en uno solo, con una nueva columna pais.
- Elimina duplicados, nulos, columnas sin información útil (video_id, thumbnail_link, description, video_error_or_removed) y las categorías con muy pocos vídeos (Trailers, Movies, Nonprofits & Activism).
- Deja un único registro por vídeo y país (el de más views), ya que un mismo vídeo aparece en tendencias varios días.
- Crea dias_desde_publicacion: días desde que se publicó el vídeo hasta que fue tendencia por primera vez.
- Limpia los espacios de title, channel_title y tags.

**2. Entrenamiento**

```
python train_models.py train.conf
```

Lee de [train.conf](train.conf) la ruta de los datos limpios (sección `[datos]`) y la ruta donde guardar el pickle de cada modelo (una sección por modelo). Las rutas relativas se toman desde la carpeta de train.conf. Entrena cada modelo con `fit` y lo guarda con `save`.

**3. Inferencia**

```
python inference_model.py <modelo> <pesos.pkl> <datos.csv>
python inference_model.py rf_classifier pesos/rf_classifier.pkl nuevos_videos.csv
```

- `modelo`: `rf_classifier`, `rf_regressor`, `kmeans_clustering` o `knn_recommender`.
- `pesos`: el pickle guardado por train_models.py.
- `datos`: fichero .csv o .pkl con las mismas columnas que los datos limpios. Para el recomendador basta con una columna title con los títulos de los vídeos para los que queremos recomendaciones.

Muestra los datos con una columna `prediccion` (categoría, likes o cluster). En el recomendador, muestra los vídeos recomendados para cada título.

## 🤖 Modelo

Cada clase tiene los métodos públicos `__init__`, `fit`, `predict`, `save` y `load`, y un método privado `__preprocess_data` con las transformaciones propias del modelo. Esas transformaciones se ajustan en `fit` y se guardan en el pickle junto al modelo. `predict` acepta un DataFrame, un numpy array, una Series o un diccionario, así que sirve tanto para una observación como para un conjunto.

| Problema | Clase | Modelo elegido | Transformaciones | Resultado en test |
|---|---|---|---|---|
| Categoría | `RFClassifier` | Random Forest optimizado con RandomizedSearchCV | One-hot de pais | F1 macro 0,32, accuracy 0,39 |
| Likes | `RFRegressor` | Random Forest con parámetros por defecto (el optimizado salió algo peor) | One-hot de pais y category (sin logaritmo ni escalado: los árboles no lo necesitan) | R2 0,949, MAE ≈ 4.900 likes |
| Clusterización | `KMeansClustering` | K-Means con k = 4 (codo y silhouette) | Nuevas variables (n_tags, longitud_titulo, n_videos_canal), logaritmo, escalado y PCA a 2 componentes | Silhouette 0,35 |
| Recomendación | `KNNRecommender` | KNN con distancia del coseno sobre TF-IDF | Título, tags, canal y categoría unidos en un solo texto | 72 % de recomendaciones de la misma categoría (15 % al azar) |

## 📓 Notebook

En [exploratory_analysis.ipynb](exploratory_analysis.ipynb) se hace el análisis exploratorio de los datos y, para cada problema, se comparan los modelos que aplican:

- **Categoría:** Regresión Logística, Random Forest, KNN y SVM. El SVM se limita a 1.000 iteraciones para que no tarde horas, así que no llega a converger y sale el peor; la Regresión Logística usa `max_iter=1000` porque con las 100 por defecto no convergía.
- **Reto opcional (comentarios o ratings desactivados):** Regresión Logística, Random Forest, KNN y SVM. Se estudia con la matriz de confusión el sesgo hacia la clase mayoritaria y se mitiga con GridSearchCV y `class_weight="balanced"`.
- **Likes:** Regresión Lineal, Perceptrón Multicapa y Random Forest.
- **Clusterización:** K-Means, DBSCAN, Gaussian Mixture Models y HDBSCAN, con PCA, t-SNE, LDA y un autoencoder (MLPRegressor) como técnicas de reducción de dimensionalidad. Se clusteriza sobre PCA y sobre el autoencoder para comparar ambas reducciones.
- **Recomendación:** TF-IDF con KNN.

Con el mejor modelo de cada problema se buscan los hiperparámetros óptimos, y esos son los modelos industrializados en el paquete models.

## 💡 Conclusiones

- **Likes:** se predicen muy bien (R2 0,949).
- **Categoría:** es el problema más difícil (F1 macro 0,32, accuracy 0,39).
- **Clusterización:** K-Means con 4 grupos da la mejor separación (silhouette 0,35).
- **Recomendación:** el 72 % de los vídeos recomendados son de la misma categoría que el original, frente al 15 % si se eligieran al azar.
