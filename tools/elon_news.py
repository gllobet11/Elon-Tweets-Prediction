import os
from google.cloud import bigquery
import pandas as pd


def fetch_gdelt_data():
    # 1. Configuración del Cliente
    # BigQuery buscará tus credenciales en el entorno automáticamente
    # Asegúrate de tener tu archivo JSON de credenciales o estar logueado
    client = bigquery.Client()

    # 2. Tu Consulta SQL (La versión corregida)
    query = """
    SELECT
      DATE_TRUNC(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING)), WEEK) AS week_start,
      COUNT(*) AS news_volume,
      AVG(CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64)) AS avg_sentiment
    FROM
      `gdelt-bq.gdeltv2.gkg`
    WHERE
      DATE >= CAST(FORMAT_DATE('%Y%m%d000000', DATE_SUB(CURRENT_DATE(), INTERVAL 2 YEAR)) AS INT64)
      AND V2Persons LIKE '%Elon Musk%'
    GROUP BY
      week_start
    ORDER BY
      week_start ASC;
    """

    print("Ejecutando consulta en BigQuery (esto puede tomar unos segundos)...")

    # 3. Ejecutar la query y convertir a DataFrame de Pandas
    query_job = client.query(query)
    df = query_job.to_dataframe()

    # 4. Limpieza (Opcional pero recomendada)
    # Eliminar la última fila si la semana no ha terminado (para evitar datos incompletos)
    # Comparamos la fecha de la última fila con la fecha de hoy
    if not df.empty:
        last_date = df.iloc[-1]["week_start"]
        print(f"Última fecha encontrada: {last_date}")
        # Opcional: df = df[:-1] # Descomenta para borrar siempre la última fila

    # 5. Guardar a CSV
    filename = "data/raw/elon_musk_gdelt_features.csv"
    df.to_csv(filename, index=False)
    print(f"¡Éxito! Datos guardados en {filename}")
    print(df.tail())  # Muestra las últimas filas para verificar


if __name__ == "__main__":
    # Configura aquí la ruta a tu llave si no usas variables de entorno
    # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "ruta/a/tu-llave-secreta.json"

    fetch_gdelt_data()
