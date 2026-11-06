import argparse
import pandas as pd

#Ejemplo de uso: python preprocess_data.py . datos_limpios.csv

parser = argparse.ArgumentParser(description="Limpieza y transformaciones comunes a todos los modelos.")
parser.add_argument("ruta_entrada", help="Carpeta con los XXvideos.csv y XX_category_id.json de cada país")
parser.add_argument("ruta_salida", help="CSV donde se guardan los datos limpios")
args = parser.parse_args()


def actualizacion_categoria_pais(ruta_csv, ruta_json, nombre_pais):
    """Función que actualiza la category_id del Csv reemplazando por el valor string asignado
    en el fichero json y añade columna con el pais."""

    dfjson = pd.read_json(ruta_json)#en vez de leer linea por linea, empaqueta todo el json
    dfjson_aplanado = pd.json_normalize(dfjson["items"])
    dfjson_items = pd.DataFrame({"json_id": dfjson_aplanado["id"].astype(str),
                                 "category": dfjson_aplanado["snippet.title"]})

    dfcsv = pd.read_csv(ruta_csv, encoding="utf-8", encoding_errors="replace")
    dfcsv["category_id"] = dfcsv["category_id"].astype(str)

    df_final = pd.merge(dfcsv, dfjson_items, left_on="category_id", right_on="json_id", how="left")

    df_final = df_final.drop(columns=["json_id", "category_id"])

    df_final["pais"] = nombre_pais#creamos la columna pais

    return df_final


paises = {"CA": "Canadá", "DE": "Alemania", "FR": "Francia", "GB": "Gran Bretaña", "IN": "India",
          "JP": "Japón", "KR": "Corea", "MX": "Mexico", "RU": "Rusia", "US": "USA"}

dataframe_por_pais = (actualizacion_categoria_pais(f"{args.ruta_entrada}/{c}videos.csv",
                                                   f"{args.ruta_entrada}/{c}_category_id.json", nombre)
                      for c, nombre in paises.items())

dataframe_final = pd.concat(dataframe_por_pais, ignore_index=True)

dataframe_final.drop_duplicates(inplace=True)
dataframe_final.drop(["video_id", "thumbnail_link", "description"], axis=1, inplace=True)
dataframe_final.dropna(inplace=True)

dataframe_final["trending_date"] = pd.to_datetime(dataframe_final["trending_date"],
                                                  format="%y.%d.%m", errors="coerce").dt.floor("D")
primer_trending = dataframe_final.groupby(["channel_title", "title", "pais"])[
    "trending_date"].transform("min")

dataframe_final.sort_values(by="views", ascending=False, inplace=True)
dataframe_final.drop_duplicates(subset=["channel_title", "title", "pais"],keep="first",inplace=True)

dataframe_final["publish_time"] = pd.to_datetime(dataframe_final["publish_time"],
                                                 errors="coerce").dt.tz_localize(None).dt.floor("D")

dataframe_final["dias_desde_publicacion"] = (primer_trending.loc[dataframe_final.index]
                                             - dataframe_final["publish_time"]).dt.days.clip(
                                                 lower=0)

dataframe_final.drop(["trending_date", "publish_time"], axis=1, inplace=True)

dataframe_final["title"] = (dataframe_final["title"].str.strip().str.split().str.join(" "))
dataframe_final["channel_title"] = (dataframe_final["channel_title"].str.strip().str.split()
                                    .str.join(" "))

dataframe_final["tags"] = (dataframe_final["tags"].str.strip().str.split().str.join(" "))
dataframe_final["tags"] = dataframe_final["tags"].str.replace('"', "", regex=False)
dataframe_final["tags"] = (dataframe_final["tags"].str.strip().str.split("|", regex=False))


def eliminar_etiquetas_vacias(lista_de_tags):
    """Filtra y elimina los elementos vacíos de la lista de etiquetas."""
    resultado = []
    for tag in lista_de_tags:
        if tag != "":
            resultado.append(tag)
    return resultado


dataframe_final["tags"] = dataframe_final["tags"].apply(eliminar_etiquetas_vacias)

dataframe_final = dataframe_final[~dataframe_final["category"].isin(["Trailers", "Movies", "Nonprofits & Activism"])]
dataframe_final = dataframe_final.drop("video_error_or_removed", axis=1)

dataframe_final["tags"] = dataframe_final["tags"].str.join("|")
dataframe_final.to_csv(args.ruta_salida, index=False)
