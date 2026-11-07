# 📺 YouTube Trending Videos

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) ![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

Práctica final de IA con los vídeos en tendencia de YouTube de 10 países: predecir la categoría y los likes de un vídeo, agrupar los vídeos y recomendar vídeos parecidos.

## 🗂️ Estructura

```
├── exploratory_analysis.ipynb   Análisis exploratorio, comparación de modelos y conclusiones
├── preprocess_data.py           Limpieza de los datos en crudo
├── train_models.py              Entrena los modelos y los guarda
├── inference_model.py           Usa un modelo guardado con datos nuevos
├── train.conf                   Rutas de los datos y de los modelos
└── models/                      Una clase por modelo (fit, predict, save, load)
```

## ▶️ Uso

Con Python 3, pandas y scikit-learn (por ejemplo, Anaconda), desde la carpeta del proyecto:

```
python preprocess_data.py . datos_limpios.csv
python train_models.py train.conf
python inference_model.py rf_classifier rf_classifier.pkl nuevos_videos.csv
```

En la inferencia, el modelo puede ser `rf_classifier`, `rf_regressor`, `kmeans_clustering` o `knn_recommender`. Para el recomendador basta con indicar el título, el canal o la categoría de los vídeos.

## 🤖 Modelos y resultados

| Problema | Modelo | Resultado |
|---|---|---|
| Categoría | Random Forest optimizado | F1 macro 0,32 |
| Likes | Random Forest | R2 0,949 |
| Grupos de vídeos | K-Means (2 grupos) sobre PCA | Silhouette 0,41 |
| Recomendación | Vídeos más parecidos en título, tags y canal, sin repetir versiones del mismo vídeo | Recomienda la misma canción o el mismo artista |

Los likes se predicen muy bien; la categoría es lo más difícil. Los grupos separan los vídeos más populares de los menos populares, sin relación con las categorías.
