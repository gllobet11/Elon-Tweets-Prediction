import pandas as pd
import numpy as np
from prophet import Prophet

# ---------------------------------------------------------
# 1. CARGA Y PREPARACIÓN DE DATOS DE NOTICIAS (GDELT)
# ---------------------------------------------------------
df_news = pd.read_csv("data/raw/elon_musk_gdelt_features.csv")

# Convertir a datetime y eliminar la zona horaria (Prophet prefiere tz-naive)
df_news["ds"] = pd.to_datetime(df_news["week_start"]).dt.tz_localize(None)

# IMPORTANTE: Eliminar la última fila si es la semana actual (incompleta)
# En tu output, la fila 103 tiene solo 89 noticias vs 2000 de las anteriores.
# Si no la borras, el modelo pensará que el interés cayó a 0.
if not df_news.empty:
    df_news = df_news.iloc[:-1]

# Transformación Logarítmica para suavizar picos virales
df_news["news_vol_log"] = np.log1p(df_news["news_volume"])

# CREACIÓN DE FEATURES "LAGGED" (El secreto del Leading Indicator)
# Movemos los datos 1 semana hacia adelante.
# Significado: El volumen de noticias de la semana T se alinea con la fecha T+1
df_news["news_vol_lag1"] = df_news["news_vol_log"].shift(1)
df_news["sentiment_lag1"] = df_news["avg_sentiment"].shift(1)

# Seleccionamos solo lo que necesitamos para el merge
df_features = df_news[["ds", "news_vol_lag1", "sentiment_lag1"]].dropna()

# ---------------------------------------------------------
# 2. INTEGRACIÓN CON TU DATASET PRINCIPAL (TWEETS)
# ---------------------------------------------------------
# Supongamos que 'df_tweets' es tu dataframe original con columnas ['ds', 'y']
# Asegúrate de que df_tweets['ds'] también esté alineado a DOMINGOS.

# Ejemplo simulado de tu df_tweets (Bórralo cuando uses tu data real)
# df_tweets = pd.read_csv('tus_tweets.csv')
# df_tweets['ds'] = pd.to_datetime(df_tweets['ds'])

# HACEMOS EL MERGE
# Usamos 'left' para mantener todas las fechas de tweets,
# rellenando con noticias donde haya coincidencia.
df_final = pd.merge(df_tweets, df_features, on="ds", how="left")

# Relleno de nulos (Crucial para Prophet)
# Si no hay noticias (ej. al inicio del histórico), rellenamos con 0 o la media
df_final["news_vol_lag1"] = df_final["news_vol_lag1"].fillna(0)
df_final["sentiment_lag1"] = df_final["sentiment_lag1"].fillna(0)

# ---------------------------------------------------------
# 3. CONFIGURACIÓN DEL MODELO PROPHET
# ---------------------------------------------------------

model = Prophet(
    weekly_seasonality=True,
    yearly_seasonality=True,
    # Otros hiperparámetros que ya tengas...
)

# Añadimos los Regresores Externos
model.add_regressor("news_vol_lag1")
model.add_regressor("sentiment_lag1")

# Entrenamiento
model.fit(df_final)

# ---------------------------------------------------------
# 4. PREDICCIÓN (FUTURE DATAFRAME)
# ---------------------------------------------------------
future = model.make_future_dataframe(periods=12, freq="W-SUN")  # 12 semanas a futuro

# Aquí está la magia:
# Como usamos LAG1, para predecir la semana que viene,
# necesitamos las noticias de ESTA semana (que ya las tenemos).
# Debemos unir las features al dataframe 'future' también.

future = pd.merge(future, df_features, on="ds", how="left")

# Si hay huecos futuros donde aún no ha pasado la noticia (más allá de 1 semana),
# Prophet fallará. Una estrategia común es usar el último valor conocido
# o el promedio para futuros lejanos.
future["news_vol_lag1"] = future["news_vol_lag1"].ffill()  # Forward Fill
future["sentiment_lag1"] = future["sentiment_lag1"].ffill()

forecast = model.predict(future)

# Visualización rápida de componentes para ver si la noticia impacta
model.plot_components(forecast)
