import argparse
import os

import pandas as pd

#Ejemplo de uso: python preprocess_data.py . datos_limpios.csv

parser = argparse.ArgumentParser(description="Limpieza y transformaciones comunes a todos los modelos.")
parser.add_argument("ruta_entrada", help="carpeta con los XXvideos.csv y XX_category_id.json de cada país")
parser.add_argument("ruta_salida", help="csv donde se guardan los datos limpios")
args = parser.parse_args()

paises = {"CA": "Canadá", "DE": "Alemania", "FR": "Francia", "GB": "Gran Bretaña", "IN": "India",
          "JP": "Japón", "KR": "Corea", "MX": "Mexico", "RU": "Rusia", "US": "USA"}


def leer_pais(codigo, nombre):
    """Lee los vídeos de un país, cambia category_id por el nombre de su categoría (del json) y añade el país."""
    items = pd.json_normalize(pd.read_json(os.path.join(args.ruta_entrada, f"{codigo}_category_id.json"))["items"])
    categorias = pd.DataFrame({"json_id": items["id"].astype(str), "category": items["snippet.title"]})
    df = pd.read_csv(os.path.join(args.ruta_entrada, f"{codigo}videos.csv"), encoding="utf-8", encoding_errors="replace")
    df["category_id"] = df["category_id"].astype(str)
    df = df.merge(categorias, left_on="category_id", right_on="json_id", how="left").drop(columns=["json_id", "category_id"])
    df["pais"] = nombre
    return df


df = pd.concat([leer_pais(codigo, nombre) for codigo, nombre in paises.items()], ignore_index=True)
df = df.drop_duplicates().drop(columns=["video_id", "thumbnail_link", "description", "video_error_or_removed"])
df = df.dropna()#los vídeos sin categoría
df = df[~df["category"].isin(["Trailers", "Movies", "Nonprofits & Activism"])]#categorías con muy pocos vídeos

#primer día en tendencias de cada vídeo en cada país, antes de quitar sus repeticiones
df["trending_date"] = pd.to_datetime(df["trending_date"], format="%y.%d.%m", errors="coerce").dt.floor("D")
df["primer_trending"] = df.groupby(["channel_title", "title", "pais"])["trending_date"].transform("min")

#un registro por vídeo y país: el de más views
df = df.sort_values(by="views", ascending=False).drop_duplicates(subset=["channel_title", "title", "pais"])

#días desde la publicación hasta su primer día en tendencias
publicacion = pd.to_datetime(df["publish_time"], errors="coerce").dt.tz_localize(None).dt.floor("D")
df["dias_desde_publicacion"] = (df["primer_trending"] - publicacion).dt.days.clip(lower=0)
df = df.drop(columns=["trending_date", "publish_time", "primer_trending"])

#quitamos espacios y saltos de línea de más
for columna in ["title", "channel_title", "tags"]:
    df[columna] = df[columna].str.split().str.join(" ")

#tags sin comillas ni etiquetas vacías, separados por "|" (un csv no guarda listas; las clases de models los separan)
df["tags"] = (df["tags"].str.replace('"', "", regex=False).str.strip().str.split("|", regex=False)
              .apply(lambda tags: "|".join(tag for tag in tags if tag != "")))

df.to_csv(args.ruta_salida, index=False)
