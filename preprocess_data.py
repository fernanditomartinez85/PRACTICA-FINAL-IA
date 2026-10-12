import os
import pandas as pd

#TANTO LOS ARCHIVOS BRUTOS COMO EL ARCHIVO LIMPIO SE ENCUENTRAN EN LA MISMA CARPETA QUE CONTIENE ESTE SCRIPT

ruta_entrada = os.path.dirname(os.path.abspath(__file__))
ruta_salida = os.path.join(ruta_entrada, "dataframe_final_limpio.pkl")


def actualizacion_categoria_pais(ruta_csv, ruta_json, nombre_pais):

    """Función que actualiza la category_id del Csv reemplazando por el valor string asignado en el fichero json y añade columna con el pais."""

    dfjson = pd.read_json(ruta_json)#en vez de leer linea por linea, empaqueta todo el json
    dfjson_aplanado = pd.json_normalize(dfjson['items'])
    dfjson_items = pd.DataFrame({'json_id': dfjson_aplanado['id'].astype(str),'category': dfjson_aplanado['snippet.title']})

    dfcsv = pd.read_csv(ruta_csv, encoding='utf-8', encoding_errors='replace')
    dfcsv['category_id'] = dfcsv['category_id'].astype(str)

    df_final = pd.merge(dfcsv, dfjson_items, left_on='category_id', right_on='json_id', how='left')

    df_final = df_final.drop(columns=['json_id', 'category_id'])

    df_final['pais'] = nombre_pais#creamos la columna pais

    return df_final


paises = {"CA":"Canadá","DE":"Alemania","FR":"Francia","GB":"Gran Bretaña","IN":"India","JP":"Japón","KR":"Corea","MX":"Mexico","RU":"Rusia","US":"USA"}

dataframe_por_pais = (actualizacion_categoria_pais(os.path.join(ruta_entrada, f"{c}videos.csv"), os.path.join(ruta_entrada, f"{c}_category_id.json"), nombre) for c, nombre in paises.items())

#Unificamos todos los dataframes
dataframe_final = pd.concat(dataframe_por_pais, ignore_index=True)

#Quitamos duplicados exactos
dataframe_final.drop_duplicates(inplace=True)

#Eliminamos columnas sin información útil para los modelos
dataframe_final.drop(["video_id", "thumbnail_link", "description", "video_error_or_removed"], axis=1, inplace=True)

#Eliminamos los nulos de category
dataframe_final.dropna(inplace=True)

#Eliminamos las categorías con muy pocos registros
dataframe_final.drop(dataframe_final[dataframe_final["category"].isin(["Trailers", "Movies", "Nonprofits & Activism"])].index, inplace=True)

#Antes de quitar duplicados guardamos el primer día en tendencias de cada vídeo en cada país
dataframe_final["trending_date"] = pd.to_datetime(dataframe_final["trending_date"], format="%y.%d.%m", errors="coerce").dt.floor('D')
dataframe_final["primer_trending"] = dataframe_final.groupby(["channel_title", "title", "pais"])["trending_date"].transform("min")

#Dejamos un único registro por video y país, el de más views
dataframe_final.sort_values(by="views", ascending=False, inplace=True)
dataframe_final.drop_duplicates(subset=["channel_title", "title", "pais"],keep="first",inplace=True)

#Calculamos los días desde la publicación hasta que fue tendencia por primera vez
dataframe_final["publish_time"] = pd.to_datetime(dataframe_final["publish_time"], errors="coerce").dt.tz_localize(None).dt.floor('D')
dataframe_final["dias_desde_publicacion"] = (dataframe_final["primer_trending"] - dataframe_final["publish_time"]).dt.days.clip(lower=0)

dataframe_final.drop(["trending_date","publish_time","primer_trending"], axis=1, inplace=True)

#Quitamos espacios y saltos de línea en title y channel_title
dataframe_final["title"] = (dataframe_final["title"].str.strip().str.split().str.join(" "))
dataframe_final["channel_title"] = (dataframe_final["channel_title"].str.strip().str.split().str.join(" "))

#En tags además separamos por "|" y lo guardamos como lista
dataframe_final["tags"] = (dataframe_final["tags"].str.strip().str.split().str.join(" "))
dataframe_final["tags"] = dataframe_final["tags"].str.replace('"', '', regex=False)#regex=False: busca el texto tal cual, sin tratarlo como expresión regular
dataframe_final["tags"] = (dataframe_final["tags"].str.strip().str.split("|", regex=False))
dataframe_final["tags"] = dataframe_final["tags"].apply(lambda lista: [tag for tag in lista if tag != ""])#quitamos las etiquetas vacías

dataframe_final.to_pickle(ruta_salida)
