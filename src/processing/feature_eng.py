import pandas as pd
import numpy as np
from datetime import timedelta

class FeatureEngineer:
    def __init__(self):
        # Parámetros por defecto detectados en tu notebook
        self.burst_quantile = 0.8
        self.regime_lookback_weeks = 52 # Para no buscar cambios de régimen hace 10 años
        
    def _calculate_regime_change(self, daily_series):
        """
        Detecta dinámicamente el cambio de régimen (caída brusca de tweets)
        replicando la lógica de la Celda 12 del notebook.
        """
        weekly = daily_series.resample("W-MON").sum()
        # Calculamos la diferencia semana a semana
        diffs = weekly.diff()
        
        # Encontramos la semana con la caída más grande (el 'break')
        # Limitamos la búsqueda al último año para relevancia
        recent_diffs = diffs.tail(self.regime_lookback_weeks)
        
        if recent_diffs.empty:
            return daily_series.index.min() # Fallback al inicio si no hay datos

        break_week = recent_diffs.idxmin()
        
        # MANEJO DE NaT: Si idxmin devuelve NaT (ej. todos los diffs son NaN),
        # usamos el inicio de la serie como fecha de cambio de régimen.
        if pd.isna(break_week):
            regime_change_date = daily_series.index.min()
        else:
            # El cambio de régimen se define como 7 días antes del break
            regime_change_date = (break_week - timedelta(days=7)).normalize()
        return regime_change_date

    def build_external_features(self, df_raw):
        """
        Calcula features de comportamiento (sueño/horarios).
        Replica la Celda 31.
        """
        df = df_raw.copy()
        
        # Asegurar UTC y nombre de columna (manejo robusto de timezone)
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
            if df['created_at'].dt.tz is None:
                # Si es naive, localizar a UTC
                df['created_at_utc'] = df['created_at'].dt.tz_localize('UTC')
            else:
                # Si ya tiene tz, solo convertir a UTC
                df['created_at_utc'] = df['created_at'].dt.tz_convert('UTC')
        
        # Agrupar por día
        daily = df.groupby(df["created_at_utc"].dt.floor("D")).agg(
            night_tweets=("created_at_utc", lambda s: ((s.dt.hour >= 0) & (s.dt.hour < 5)).sum()),
            last_tweet_hour=("created_at_utc", lambda s: s.dt.hour.max())
        )
        
        # Frecuencia diaria completa
        daily = daily.asfreq("D", fill_value=0)
        
        # Asegurar que el índice de 'daily' sea timezone-naive antes de retornar
        daily.index = daily.index.tz_localize(None)

        # Features de sueño
        daily["sleep_debt"] = daily["night_tweets"].rolling(window=3).sum().shift(1)
        daily["sleep_debt_avg"] = daily["sleep_debt"] / 3.0
        
        # Zombie Mode (basado en mediana histórica)
        median_debt = daily["sleep_debt_avg"].median()
        daily["zombie_mode"] = (daily["sleep_debt_avg"] > median_debt).astype(int)
        
        # Late Night Flag
        daily["last_tweet_hour"] = daily["last_tweet_hour"].fillna(0)
        daily["late_night_flag"] = (daily["last_tweet_hour"] >= 2).astype(int).shift(1)
        
        return daily[["sleep_debt_avg", "zombie_mode", "late_night_flag"]]

    def process_data(self, df_raw):
        """
        Pipeline principal: Raw DF -> Features procesadas listas para el modelo.
        Replica la lógica robusta del notebook.
        """
        # --- PASO 1: Crear la serie diaria CANÓNICA y CONTINUA ---
        if 'created_at' in df_raw.columns:
            df_raw['created_at'] = pd.to_datetime(df_raw['created_at']).dt.tz_localize(None) # Asegurar timezone-naive
        
        # Nuevo: Cálculo de is_reply y text_clean para reply_ratio
        df_raw['text_clean'] = df_raw['text'].astype(str).str.strip()
        df_raw['is_reply'] = df_raw['text_clean'].str.contains(r'^.{0,15} @|RT\s@', regex=True, case=False, na=False)

        df_raw['date_utc'] = df_raw['created_at'].dt.floor('D')
        
        # Crear conteos diarios incluyendo reply_count y hour_std
        daily_counts = (
            df_raw.groupby('date_utc').agg(
                n_tweets=('id', 'count'),
                reply_count=('is_reply', 'sum'),
                hour_std=('created_at', lambda x: x.dt.hour.std())
            )
            .sort_index()
        )
        
        # PASO CRÍTICO: Reindexado (Serie Canónica)
        # Crea un índice completo desde el primer tweet hasta el último
        full_idx = pd.date_range(start=daily_counts.index.min(), 
                                 end=daily_counts.index.max(), 
                                 freq='D')
        
        # Reindexar: Esto crea las filas faltantes y pone 0 en n_tweets, reply_count y NaN en hour_std
        daily_counts = daily_counts.reindex(full_idx, fill_value=0)
        # Los días reindexados que no tenían tweets tendrán NaN en hour_std, los rellenamos con 0
        daily_counts['hour_std'] = daily_counts['hour_std'].fillna(0)

        # Nombrar el índice para claridad
        daily_counts.index.name = 'date'

        # --- PASO 2: Construir TODAS las features sobre la serie continua ---
        
        # Detectar Régimen (se necesita para una feature más adelante)
        regime_start = self._calculate_regime_change(daily_counts['n_tweets'])
        print(f"-> Regime Change Detected: {regime_start.date()}")
        
        df = daily_counts.copy()
        
        # Features Externas (calculadas a partir del raw y unidas) - si las hubiera
        ext_features = self.build_external_features(df_raw) # Esto ya lo tenias
        # Asegurarse de que el índice de ext_features también sea datetime-naive para la unión
        ext_features.index = ext_features.index.tz_localize(None)
        df = df.join(ext_features, how="left")
        
        # Rellenar NaNs en features externas (para días sin tweets)
        ext_feature_cols = ["sleep_debt_avg", "zombie_mode", "late_night_flag"]
        df[ext_feature_cols] = df[ext_feature_cols].fillna(0)
        
        # Features Internas (Lags, Rolling, etc.)
        df["trend"] = np.arange(len(df))
        for lag in [1, 2, 7]:
            df[f"lag_{lag}"] = df["n_tweets"].shift(lag)
            
        df["roll_mean_3"] = df["n_tweets"].rolling(3).mean().shift(1)
        df["roll_mean_7"] = df["n_tweets"].rolling(7).mean().shift(1)
        df["momentum"] = df["roll_mean_3"] - df["roll_mean_7"] # Added momentum calculation
        df["roll_std_7"]  = df["n_tweets"].rolling(7).std().shift(1)
        df["roll_sum_7"]  = df["n_tweets"].rolling(7).sum().shift(1)
        df["roll_sum_14"] = df["n_tweets"].rolling(14).sum().shift(1)
        
        df["delta_7_14"] = df["roll_sum_7"] - (df["roll_sum_14"] / 2.0)
        df["ratio_mean_3_7"] = df["roll_mean_3"] / df["roll_mean_7"]
        df["cv_7"] = df["roll_std_7"] / df["roll_mean_7"]
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df[['ratio_mean_3_7', 'cv_7']] = df[['ratio_mean_3_7', 'cv_7']].fillna(0)
        
        df["is_spike_1"] = (df["lag_1"] > df["roll_mean_7"]).astype(int)
        
        threshold = df["n_tweets"].quantile(self.burst_quantile)
        df["is_burst_today"] = (df["n_tweets"] > threshold).astype(int)
        
        df["last_burst"] = (
            df["is_burst_today"]
            .rolling(7)
            .max()
            .shift(1)
            .fillna(0)
        ).astype(int)
        
        grp = df["last_burst"].cumsum()
        df["days_since_burst"] = (
            (~df["last_burst"].astype(bool))
            .groupby(grp)
            .cumsum()
        ).shift(1).fillna(0)
        
        df["dow"] = df.index.dayofweek
        df["post_regime"] = (df.index >= regime_start).astype(int)
        
        # Nuevas features: reply_ratio y hour_std_feature
        # Asegurarse de evitar división por cero en reply_ratio y rellenar NaN de shifts
        df['reply_ratio'] = (df['reply_count'] / df['n_tweets'].replace(0, 1)).shift(1)
        df['hour_std_feature'] = df['hour_std'].shift(1)
        
        # Rellenar NaNs para las nuevas características antes del final dropna
        df['reply_ratio'] = df['reply_ratio'].fillna(0)
        df['hour_std_feature'] = df['hour_std_feature'].fillna(0)

        print(f"DEBUG: Full features shape before final dropna: {df.shape}")
        print(f"DEBUG: Max date before final dropna: {df.index.max()}")
        
        # Rellenar los NaNs iniciales resultantes de lags/rolling con 0 (general)
        df = df.fillna(0)
        
        final_df = df.dropna(subset=['roll_sum_14'])
        print(f"DEBUG: Max date after final dropna: {final_df.index.max()}")

        return final_df


    def get_latest_features(self, df_raw):
        """
        Retorna la ÚLTIMA fila de features procesadas, filtrando solo las relevantes para el modelo.
        """
        full_features = self.process_data(df_raw)
        
        # Las features que el modelo de inferencia espera son: 'const', 'lag_1', 'last_burst'
        # Aquí solo nos encargamos de 'lag_1' y 'last_burst'. 'const' se añade en inference.py
        model_required_features = ['lag_1', 'last_burst']
        
        # Filtramos el DataFrame para que solo contenga las columnas que el modelo necesita
        # Esto evita que `dropna` elimine filas por NaNs en columnas irrelevantes.
        filtered_features = full_features[model_required_features].dropna()

        if filtered_features.empty:
            raise ValueError("❌ No hay suficientes datos limpios para generar las features necesarias para el modelo.")

        return filtered_features.iloc[[-1]]