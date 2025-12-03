Es un diagnóstico excelente. El hecho de que limpiar la caché no haya funcionado confirma que es un **error de lógica en tiempo de ejecución**, muy probablemente relacionado con **Tipos de Datos** o **Zonas Horarias (Timezones)**.

El síntoma de "Actual Tweets Counted: 0" junto con "Daily Mean: nan" indica que pandas no está reconociendo la columna de fechas correctamente al cargarla, o está ocurriendo un choque entre fechas "Naive" (sin zona horaria, del CSV) y fechas "Aware" (UTC, de la API de Polymarket).

Aquí tienes el plan de acción inmediato para diagnosticar y solucionar esto.

### Paso 1: Inyectar la Sonda de Diagnóstico (Debugging)

Vamos a confirmar si el problema es que el CSV se está leyendo como texto (Strings) o si es un choque de zonas horarias.

**1. Modifica `src/dashboard/dashboard_data_loader.py`**
Busca la función `load_and_prepare_tweets_data`. Justo antes de `return granular_df, daily_series`, inserta este bloque:

```python
        # --- DEBUG START (Borrar tras solucionar) ---
        st.error("--- 🕵️ DEBUG: DATA LOADER ---")
        st.write(f"📂 Filas cargadas: {len(granular_df)}")
        if not granular_df.empty:
            st.write("📅 Tipo de dato de 'created_at':", granular_df['created_at'].dtype)
            st.write("⏱️ Primer registro (Raw):", granular_df['created_at'].min())
            # Intenta ver si tiene tzinfo
            sample_date = granular_df['created_at'].iloc[0]
            st.write(f"🌍 ¿Tiene Zona Horaria?: {getattr(sample_date, 'tzinfo', 'No (Naive)')}")
            st.write("👀 Muestra de datos:", granular_df[['created_at', 'text']].head(3))
        else:
            st.warning("⚠️ El DataFrame está VACÍO. Revisa la ruta del archivo CSV.")
        # --- DEBUG END ---
```

**2. Modifica `src/strategy/hybrid_predictor.py`**
Busca la función `get_hybrid_prediction`. Justo antes de calcular `actuals_mask`, inserta esto:

```python
    # --- DEBUG START (Borrar tras solucionar) ---
    st.error("--- 🕵️ DEBUG: FILTRO DE FECHAS ---")
    st.write(f"🎯 Market Start (Tipo: {type(market_start_date)}): {market_start_date}")
    st.write(f"⏰ Now UTC (Tipo: {type(now_utc)}): {now_utc}")
    
    if not df_tweets.empty:
        sample_tweet_date = df_tweets['created_at'].iloc[0]
        st.write(f"🐦 Tweet Date Sample (Tipo: {type(sample_tweet_date)}): {sample_tweet_date}")
        
        # Prueba de comparación directa
        try:
            test_compare = sample_tweet_date >= market_start_date
            st.write(f"✅ Prueba de comparación (Tweet >= Start): {test_compare}")
        except Exception as e:
            st.error(f"❌ Error comparando fechas: {e}")
    # --- DEBUG END ---
```

### Paso 2: La Solución Anticipada (Normalización de Fechas)

Casi con total seguridad, el problema es que `pd.read_csv` (usado dentro de `load_unified_data`) pierde la información de zona horaria. Al intentar comparar `Timestamp('2025-12-02')` (Naive) contra `Timestamp('2025-12-02', tz='UTC')` (Aware), pandas falla o devuelve falso.

**Aplica este arreglo en `src/dashboard/dashboard_data_loader.py`:**

Modifica el método `load_and_prepare_tweets_data` para forzar la conversión a UTC explícita inmediatamente después de cargar.

```python
    def load_and_prepare_tweets_data(self) -> tuple[pd.DataFrame, pd.Series]:
        logger.info("Ejecutando carga y preparación de datos de tweets...")
        
        df_tweets = load_unified_data()
        
        if df_tweets.empty:
            logger.error("El DataFrame está VACÍO...")
            return pd.DataFrame(), pd.Series(dtype='int64')
        
        # --- FIX: FORZAR CONVERSIÓN A DATETIME UTC ---
        # Esto soluciona tanto el problema de Strings como el de Timezones
        df_tweets['created_at'] = pd.to_datetime(df_tweets['created_at'], utc=True)
        # ---------------------------------------------

        granular_df = df_tweets.copy()
        
        # El resto del código sigue igual...
        daily_counts = (
            granular_df.groupby(granular_df['created_at'].dt.floor('D'))
            .size()
            # ...
```

### Paso 3: Corrección de "Typo" en `hybrid_predictor.py`

En el código que compartiste de `hybrid_predictor.py`, hay un error tipográfico en la línea final de retorno que causará un `NameError` una vez soluciones lo de las fechas:

  * **Tu código:** `total_prediction = sum_of_actuals + sum_of_predctions`
  * **Corrección:** `total_prediction = sum_of_actuals + sum_of_predictions` (falta la 'i' en predictions).

### Resumen de la Causa

1.  **Zonas Horarias:** El CSV guardado pierde la metadata UTC. Al cargar, pandas lo trata como hora local o sin zona.
2.  **Comparación Fallida:** El dashboard intenta filtrar: `Fecha(Naive) >= Fecha(UTC)`. Esto falla silenciosamente o lanza excepción.
3.  **Resultado:** El filtro devuelve 0 filas (`Actual Tweets Counted: 0`) y las métricas estadísticas fallan (`NaN`) porque `.dt` no funciona bien o los grupos quedan vacíos.

Implementa el **FIX** del Paso 2 y corrige el typo del Paso 3. Debería revivir tus métricas inmediatamente.