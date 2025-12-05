import csv  # Importar el módulo csv
import os
from datetime import datetime, timezone

import pandas as pd

# --- Constantes ---
EPOCH_MS = 1288834974657  # Twitter Epoch


def snowflake_to_dt_utc(snowflake_id):
    """Convierte ID de Twitter a Datetime UTC."""
    try:
        snowflake_id = int(snowflake_id)
        ms = (snowflake_id >> 22) + EPOCH_MS
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    except (ValueError, TypeError):
        return pd.NaT


class XTrackerIngestor:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_and_clean_data(self):
        """
        Carga el CSV semanal de xTracker usando un parser manual para mayor robustez.
        """
        if not os.path.exists(self.data_path):
            print(f"⚠️  Advertencia: No se encuentra el archivo: {self.data_path}")
            return pd.DataFrame()

        try:
            # 1. Parsing Manual con el módulo csv
            records = []
            with open(
                self.data_path,
                encoding="utf-8",
                errors="ignore",
            ) as infile:
                reader = csv.reader(infile)
                header = next(reader)  # Saltar el encabezado

                # Encontrar los índices de las columnas que nos interesan
                try:
                    id_index = header.index("Tweet ID")
                    content_index = header.index("Content")
                except ValueError as e:
                    print(
                        f"❌ Error: El encabezado del CSV no contiene 'Tweet ID' o 'Content'. {e}",
                    )
                    return pd.DataFrame()

                for row in reader:
                    # Asegurarse de que la fila tiene suficientes columnas
                    if len(row) > max(id_index, content_index):
                        tweet_id = row[id_index]
                        content = row[content_index]
                        records.append([tweet_id, content])

            df = pd.DataFrame(records, columns=["id", "text"])

            # 2. Limpieza de IDs
            df = df.dropna(subset=["id"])
            df["id"] = df["id"].astype(str).str.replace(r"\D", "", regex=True)
            df = df[df["id"] != ""].copy()
            df["id"] = df["id"].astype("int64")

            # 3. INGENIERÍA DE FECHA (Snowflake)
            df["created_at"] = df["id"].apply(snowflake_to_dt_utc)

            # 4. Limpieza final
            df = df.dropna(subset=["created_at"])
            df["text"] = df["text"].astype(str).str.strip().str.strip('"')

            # Retornar solo lo necesario
            return df[["id", "text", "created_at"]]

        except Exception as e:
            print(f"❌ Error crítico procesando CSV xTracker: {e}")
            return pd.DataFrame()
