import os
import sys
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

# --- Path Configuration ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.ingestion.unified_feed import load_unified_data
except (ImportError, ModuleNotFoundError):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.ingestion.unified_feed import load_unified_data

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Elon Quant Dashboard", layout="wide", initial_sidebar_state="collapsed",
)


# --- Carga de Datos ---
@st.cache_data(ttl=3600)
def load_data():
    df_tweets = load_unified_data()
    df_tweets["created_at"] = pd.to_datetime(df_tweets["created_at"]).dt.tz_convert(
        None,
    )
    return df_tweets


def get_daily_series(df_tweets):
    df = df_tweets.copy()
    df["date"] = df["created_at"].dt.floor("D")
    daily_counts = df.groupby("date").size().rename("n_tweets").to_frame()
    full_idx = pd.date_range(
        start=daily_counts.index.min(), end=daily_counts.index.max(), freq="D",
    )
    return daily_counts.reindex(full_idx, fill_value=0)


# --- UI ---
st.title("🦅 Elon Musk: Mean Reversion & Volatility Tracker")
st.markdown(
    "Dashboard diseñado para detectar **sobrecalentamiento** y oportunidades de regresión a la media.",
)

try:
    raw_tweets = load_data()
    daily_data = get_daily_series(raw_tweets)

    # --- CÁLCULOS PREVIOS ---
    # 1. Datos del Mes Actual (Para el Ancla de Media)
    current_month = datetime.now().month
    current_year = datetime.now().year
    month_data = daily_data[
        (daily_data.index.month == current_month)
        & (daily_data.index.year == current_year)
    ]
    monthly_mean = month_data["n_tweets"].mean()

    # 2. Datos Últimos 7 Días (Para Volatilidad Local)
    last_7d = daily_data.tail(7).copy()
    mean_7d = last_7d["n_tweets"].mean()
    std_7d = last_7d["n_tweets"].std()

    # KPI Personalizado: ¿Cuántos días superaron la media + 1 desviación estándar?
    # Esto indica "días de furia" recientes.
    threshold = mean_7d + std_7d
    outlier_days = last_7d[last_7d["n_tweets"] > threshold].shape[0]

    # 3. Último dato cerrado (Ayer) vs Hoy (Proyección)
    yesterday_val = daily_data["n_tweets"].iloc[-2]

    st.divider()

    # --- FILA 1: ESTRATEGIA DE REGRESIÓN A LA MEDIA ---
    st.subheader("📉 Señales de Regresión a la Media")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        # KPI: Media Mensual (El Ancla)
        st.metric(
            label="Media Diaria (Mes Actual)",
            value=f"{monthly_mean:.1f}",
            help="Este es el valor al que tiende a regresar el volumen",
        )

    with c2:
        # KPI: Desviación Actual vs Media
        # ¿Cuán lejos estamos hoy (o ayer) de lo normal?
        deviation = yesterday_val - monthly_mean
        is_overheated = deviation > (monthly_mean * 0.5)  # Si está 50% por encima

        st.metric(
            label="Desviación Ayer vs Media",
            value=f"{yesterday_val:.0f}",
            delta=f"{deviation:+.1f} sobre la media",
            delta_color="inverse" if is_overheated else "normal",
            # Inverse: Si es muy alto (rojo), indica riesgo de bajada
        )

    with c3:
        # KPI Sugerido: Días Outlier (Fatiga)
        # Si hay muchos días outlier recientes, aumenta la probabilidad de descanso
        st.metric(
            label="Días 'Outlier' (Últimos 7)",
            value=f"{outlier_days} días",
            delta="Alta Volatilidad" if outlier_days >= 2 else "Estable",
            delta_color="inverse",
            help=f"Días en la última semana que superaron {threshold:.1f} tweets (Media + 1 Std)",
        )

    with c4:
        # KPI: Volatilidad (Std Dev 7d)
        st.metric(
            label="Desviación Típica (7d)",
            value=f"{std_7d:.2f}",
            help="Cuanto más alto, más impredecible es el corto plazo",
        )

    st.divider()

    # --- VISUALIZACIÓN DE BANDAS DE BOLLINGER (CASERA) ---
    # Esto visualiza tu idea perfectamente: Media vs Desviación
    st.subheader("📊 Bandas de Actividad (30 Días)")

    chart_df = daily_data.tail(30).reset_index()
    chart_df.columns = ["Fecha", "Tweets"]

    # Calcular bandas (Media +/- 1 Std)
    chart_df["Media"] = chart_df["Tweets"].rolling(7).mean()
    chart_df["Std"] = chart_df["Tweets"].rolling(7).std()
    chart_df["Upper"] = chart_df["Media"] + chart_df["Std"]
    chart_df["Lower"] = chart_df["Media"] - chart_df["Std"]

    # Gráfico Base
    base = alt.Chart(chart_df).encode(x="Fecha:T")

    # Barras de Tweets
    bars = base.mark_bar(opacity=0.5, color="#1DA1F2").encode(
        y=alt.Y("Tweets:Q"), tooltip=["Fecha", "Tweets"],
    )

    # Línea de Media
    line_mean = base.mark_line(color="black", strokeDash=[5, 5]).encode(
        y="Media:Q", tooltip=[alt.Tooltip("Media", format=".1f", title="Media 7d")],
    )

    # Área de Desviación (Banda)
    band = base.mark_area(opacity=0.2, color="gray").encode(y="Lower:Q", y2="Upper:Q")

    # Puntos Outlier (Rojos)
    outliers = (
        base.mark_circle(color="red", size=60)
        .encode(y="Tweets:Q", tooltip=["Fecha", "Tweets"])
        .transform_filter(alt.datum.Tweets > alt.datum.Upper)
    )

    chart = (band + bars + line_mean + outliers).properties(
        height=400, title="Volumen vs Media Móvil 7d (Puntos Rojos = Outliers > 1 Std)",
    )

    st.altair_chart(chart, use_container_width=True)
    # --- PEGAR ESTO A CONTINUACIÓN DEL ÚLTIMO GRÁFICO (DENTRO DEL TRY) ---

    st.divider()
    st.subheader("📆 Análisis de Trayectoria Semanal (Crucial para Polymarket)")

    col_A, col_B = st.columns([2, 1])

    with col_A:
        # --- 1. GRÁFICO ACUMULADO (CUMULATIVE SUM) ---
        # Preparamos los datos
        # Crear columna de "Día de la Semana" (0=Mon, 6=Sun)
        df_cum = daily_data.copy()
        df_cum["weekday"] = df_cum.index.dayofweek

        # 1. Datos Semana Actual (Calculamos el inicio de la semana actual, asumiendo Lunes como inicio)
        # Ajusta 'W-MON' si tu semana de Polymarket empieza otro día (ej. Viernes)
        current_week_start = df_cum.index.max() - pd.Timedelta(
            days=df_cum.index.max().weekday(),
        )
        current_week_data = df_cum[df_cum.index >= current_week_start].copy()
        current_week_data["cumsum"] = current_week_data["n_tweets"].cumsum()
        current_week_data["Type"] = "Semana Actual"

        # 2. Datos Semana Anterior
        last_week_start = current_week_start - pd.Timedelta(days=7)
        last_week_data = df_cum[
            (df_cum.index >= last_week_start) & (df_cum.index < current_week_start)
        ].copy()
        last_week_data["cumsum"] = last_week_data["n_tweets"].cumsum()
        last_week_data["Type"] = "Semana Anterior"
        # Alinear dias para el gráfico (truco visual: ponemos misma fecha ficticia o usamos weekday)

        # 3. Promedio Histórico (Últimos 3 meses)
        three_months_ago = df_cum.index.max() - pd.DateOffset(months=3)
        hist_data = df_cum[df_cum.index >= three_months_ago].copy()
        avg_week = (
            hist_data.groupby("weekday")["n_tweets"].mean().cumsum().reset_index()
        )
        avg_week.columns = ["weekday", "cumsum"]
        avg_week["Type"] = "Promedio (3 meses)"

        # Unir para Altair
        # Mapear weekday 0-6 a nombres para el eje X
        days_map = {
            0: "Lun",
            1: "Mar",
            2: "Mié",
            3: "Jue",
            4: "Vie",
            5: "Sáb",
            6: "Dom",
        }

        combined_chart_data = pd.concat(
            [
                current_week_data[["weekday", "cumsum", "Type"]],
                last_week_data[["weekday", "cumsum", "Type"]],
                avg_week[["weekday", "cumsum", "Type"]],
            ],
        )
        combined_chart_data["Día"] = combined_chart_data["weekday"].map(days_map)

        # Gráfico de Líneas
        chart_cum = (
            alt.Chart(combined_chart_data)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "weekday",
                    title="Día de la Semana",
                    axis=alt.Axis(
                        labelExpr="datum.value == 0 ? 'Lun' : datum.value == 1 ? 'Mar' : datum.value == 2 ? 'Mié' : datum.value == 3 ? 'Jue' : datum.value == 4 ? 'Vie' : datum.value == 5 ? 'Sáb' : 'Dom'",
                    ),
                ),
                y=alt.Y("cumsum", title="Tweets Acumulados"),
                color=alt.Color(
                    "Type",
                    scale=alt.Scale(
                        domain=[
                            "Semana Actual",
                            "Semana Anterior",
                            "Promedio (3 meses)",
                        ],
                        range=["red", "gray", "blue"],
                    ),
                ),
                tooltip=["Type", "Día", alt.Tooltip("cumsum", format=".0f")],
            )
            .properties(title="Carrera Semanal: Acumulado vs Histórico", height=350)
        )

        st.altair_chart(chart_cum, use_container_width=True)

    with col_B:
        # --- 2. ESTACIONALIDAD (Heatmap Semanal) ---
        st.markdown("**Patrón Diario (Últimos 90 días)**")

        seasonality = hist_data.groupby("weekday")["n_tweets"].mean().reset_index()
        seasonality["Día"] = seasonality["weekday"].map(days_map)

        # Ordenar para el gráfico
        day_order = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]

        chart_bar = (
            alt.Chart(seasonality)
            .mark_bar()
            .encode(
                x=alt.X("n_tweets", title="Media Tweets"),
                y=alt.Y("Día", sort=day_order, title=""),
                color=alt.Color("n_tweets", legend=None),
                tooltip=["Día", alt.Tooltip("n_tweets", format=".1f")],
            )
            .properties(height=350)
        )

        st.altair_chart(chart_bar, use_container_width=True)

    # --- 3. PROYECCIÓN FINAL (Métrica de Texto) ---
    # Calculamos qué % de la semana promedio llevamos completado
    today_idx = current_week_data["weekday"].max()

    # Cuántos tweets suele hacer en los días que FALTAN
    remaining_days_avg = avg_week[avg_week["weekday"] > today_idx]

    if not remaining_days_avg.empty:
        # El acumulado total promedio menos el acumulado hasta hoy promedio (del modelo histórico)
        avg_total = avg_week["cumsum"].max()
        # Buscamos el valor acumulado promedio para el día actual
        avg_until_today_rows = avg_week[avg_week["weekday"] == today_idx]

        if not avg_until_today_rows.empty:
            avg_until_today = avg_until_today_rows["cumsum"].values[0]
            expected_remaining = avg_total - avg_until_today
        else:
            expected_remaining = 0
    else:
        expected_remaining = 0  # Semana terminada o fin de semana completo

    current_total = current_week_data["cumsum"].max()
    projected_total = current_total + expected_remaining

    st.info(f"""
    🧠 **Proyección Ingenua (Naive Projection):**
    
    Llevamos **{current_total}** tweets esta semana. Históricamente, en los días restantes Elon hace **{expected_remaining:.0f}** tweets más.
    
    👉 **Proyección Final Estimada: ~{projected_total:.0f} Tweets** (Si se comporta como el promedio reciente).
    """)

except Exception as e:
    st.error(f"Error: {e}")
