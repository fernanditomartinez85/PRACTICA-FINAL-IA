import sys
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit, explode, max as _max, to_date, to_timestamp, year, month, trim, replace, split, array_remove,date_format
from functools import reduce
import pandas as pd
import os

#TANTO LOS ARCHIVOS BRUTOS COMO EL ARCHIVO LIMPIO SE ENCUENTRAN EN LA MISMA CARPETA QUE CONTIENE ESTE SCRYPT

if os.name == "nt":  # 'nt' significa que el sistema operativo es Windows
    os.environ["HADOOP_HOME"] = os.getcwd()
    os.environ["hadoop.home.dir"] = os.getcwd()

spark=SparkSession.builder.appName("Preprocesamiento").getOrCreate()
spark.conf.set("spark.sql.ansi.enabled", "false")

def actualizacion_categoria_pais(ruta_csv, ruta_json, nombre_pais):

    """Función que actualiza la category_id del Csv reemplazando por el valor string asignado en el fichero json y añade columna con el pais"""
    
    dfjson = spark.read.option("multiLine", "true").json(ruta_json)#en vez de leer linea por linea, empaqueta todo el json
    dfjson_aplanado = dfjson.select(explode("items").alias("item")).select(col("item.id").alias("json_id"), col("item.snippet.title").alias("category"))
    
    dfcsv = spark.read.options(header=True, encoding="UTF-8").csv(ruta_csv)
    dfcsv = dfcsv.withColumn("category_id", col("category_id").cast("string"))
    
    df_final = dfcsv.join(dfjson_aplanado, dfcsv["category_id"] == dfjson_aplanado["json_id"], "left")
    
    df_final = df_final.drop("json_id","category_id")

    df_final = df_final.withColumn("pais", lit(nombre_pais))
                
    return df_final
    
paises = {"CA":"Canadá","DE":"Alemania","FR":"Francia","GB":"Gran Bretaña","IN":"India","JP":"Japón","KR":"Corea","MX":"Mexico","RU":"Rusia","US":"USA"}
#Si entrara un dataframe ES:España no nos lo leería por lo que si fuera necesario habria que automatizar el código para que lea cualquier dataframe de cualquier país. En este caso, entendemos que los ficheros siempre serán de los mismo países.

dataframes_spark = (actualizacion_categoria_pais(f"{c}videos.csv", f"{c}_category_id.json", nombre) for c, nombre in paises.items())

df_spark_unido= reduce(DataFrame.union, dataframes_spark)

df_spark_unido = df_spark_unido.dropDuplicates()

df_spark_unido = df_spark_unido.drop("video_id", "thumbnail_link","description","video_error_or_removed")

df_spark_unido = df_spark_unido.dropna()

df_max_visitas = df_spark_unido.groupBy("title", "channel_title", "pais").agg(_max("views").alias("views"))
df_spark_unido = df_spark_unido.join(df_max_visitas, on=["title", "channel_title", "pais", "views"], how="inner")

df_spark_unido = df_spark_unido \
    .withColumn("views", col("views").cast("integer")) \
    .withColumn("likes", col("likes").cast("integer")) \
    .withColumn("dislikes", col("dislikes").cast("integer")) \
    .withColumn("comment_count", col("comment_count").cast("integer")) \
    .withColumn("comments_disabled", col("comments_disabled").cast("boolean")) \
    .withColumn("ratings_disabled", col("ratings_disabled").cast("boolean")) \
    .withColumn("video_error_or_removed", col("video_error_or_removed").cast("boolean")) \
    .withColumn("trending_date", date_format(to_date(col("trending_date"), "yy.dd.MM"), "dd/MM/yyyy")) \
    .withColumn("publish_time", date_format(col("publish_time").cast("date"), "dd/MM/yyyy"))

df_spark_unido = df_spark_unido \
    .withColumn("title", replace(col("title"), lit("\n"), lit(" "))) \
    .withColumn("title", replace(col("title"), lit("\t"), lit(" "))) \
    .withColumn("title", replace(col("title"), lit('"'), lit("")))

df_spark_unido = df_spark_unido.withColumn("title", trim(col("title")))

df_spark_unido = df_spark_unido \
    .withColumn("title", replace(col("title"), lit("  "), lit(" "))) \
    .withColumn("title", replace(col("title"), lit("  "), lit(" "))) \
    .withColumn("title", replace(col("title"), lit("  "), lit(" ")))

df_spark_unido = df_spark_unido \
    .withColumn("channel_title", replace(col("channel_title"), lit("\n"), lit(" "))) \
    .withColumn("channel_title", replace(col("channel_title"), lit("\t"), lit(" "))) \
    .withColumn("channel_title", replace(col("channel_title"), lit('"'), lit("")))

df_spark_unido = df_spark_unido.withColumn("channel_title", trim(col("channel_title")))

df_spark_unido = df_spark_unido \
    .withColumn("channel_title", replace(col("channel_title"), lit("  "), lit(" "))) \
    .withColumn("channel_title", replace(col("channel_title"), lit("  "), lit(" "))) \
    .withColumn("channel_title", replace(col("channel_title"), lit("  "), lit(" "))) 

df_spark_unido = df_spark_unido.withColumn("tags", replace(col("tags"), lit("|"), lit(" "))) \
    .withColumn("tags", replace(col("tags"), lit("\n"), lit(" "))) \
    .withColumn("tags", replace(col("tags"), lit("\t"), lit(" "))) \
    .withColumn("tags", replace(col("tags"), lit('"'), lit("")))

df_spark_unido = df_spark_unido.withColumn("tags", trim(col("tags")))

df_spark_unido = df_spark_unido \
    .withColumn("tags", replace(col("tags"), lit("  "), lit(" "))) \
    .withColumn("tags", replace(col("tags"), lit("  "), lit(" "))) \
    .withColumn("tags", replace(col("tags"), lit("  "), lit(" ")))

df_spark_unido = df_spark_unido.withColumn("tags", array_remove(split(col("tags"), " "), "")) 

Ruta_limpia = ""
ruta_salida_final = f"{Ruta_limpia}Dataset_limpio.parquet"
df_spark_unido.toPandas().to_parquet(ruta_salida_final, index=False)

spark.stop()