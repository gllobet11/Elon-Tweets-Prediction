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
    from src.dashboard.dashboard_data_loader import DashboardDataLoader
except (ImportError, ModuleNotFoundError):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.dashboard.dashboard_data_loader import DashboardDataLoader

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Elon Quant Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --- Carga de Datos ---
@st.cache_data(ttl=900)  # Cache for 15 minutes
def load_all_data():
    """
    Uses the new loader to get all necessary dataframes.
    """
    loader = DashboardDataLoader()
    # granular_data is now in ET, daily_data is also in ET
    granular_data, daily_data = loader.load_and_prepare_tweets_data()
    return granular_data, daily_data


# --- UI ---
st.title("🦅 Elon Musk: Mean Reversion & Volatility Tracker")
st.markdown(
    "Dashboard diseñado para detectar **sobrecalentamiento** y oportunidades de regresión a la media.",
)

try:
    granular_data_et, daily_data = load_all_data() # daily_data is already in ET

    # --- CÁLCULOS PREVIOS ---
    # The index is already timezone-aware (America/New_York)
    # 1. Datos del Mes Actual (Para el Ancla de Media)
    now_et = datetime.now(daily_data.index.tz)
    current_month = now_et.month
    current_year = now_et.year
    
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
    threshold = mean_7d + std_7d
    outlier_days = last_7d[last_7d["n_tweets"] > threshold].shape[0]

    # 3. Último dato cerrado (Ayer) vs Hoy (Proyección)
    yesterday_val = daily_data["n_tweets"].iloc[-2]

    st.divider()

    # --- FILA 1: ESTRATEGIA DE REGRESIÓN A LA MEDIA ---
    st.subheader("📉 Señales de Regresión a la Media")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            label="Media Diaria (Mes Actual)",
            value=f"{monthly_mean:.1f}",
            help="Este es el valor al que tiende a regresar el volumen",
        )

    with c2:
        deviation = yesterday_val - monthly_mean
        is_overheated = deviation > (monthly_mean * 0.5)

        st.metric(
            label="Desviación Ayer vs Media",
            value=f"{yesterday_val:.0f}",
            delta=f"{deviation:+.1f} sobre la media",
            delta_color="inverse" if is_overheated else "normal",
        )

    with c3:
        st.metric(
            label="Días 'Outlier' (Últimos 7)",
            value=f"{outlier_days} días",
            delta="Alta Volatilidad" if outlier_days >= 2 else "Estable",
            delta_color="inverse",
            help=f"Días en la última semana que superaron {threshold:.1f} tweets (Media + 1 Std)",
        )

    with c4:
        st.metric(
            label="Desviación Típica (7d)",
            value=f"{std_7d:.2f}",
            help="Cuanto más alto, más impredecible es el corto plazo",
        )

    st.divider()

    # --- VISUALIZACIÓN DE BANDAS DE BOLLINGER ---
    st.subheader("📊 Bandas de Actividad (30 Días)")

    chart_df = daily_data.tail(30).reset_index()
    chart_df.columns = ["Fecha", "Tweets"]

    chart_df["Media"] = chart_df["Tweets"].rolling(7).mean()
    chart_df["Std"] = chart_df["Tweets"].rolling(7).std()
    chart_df["Upper"] = chart_df["Media"] + chart_df["Std"]
    chart_df["Lower"] = chart_df["Media"] - chart_df["Std"]

    base = alt.Chart(chart_df).encode(x="Fecha:T")
    bars = base.mark_bar(opacity=0.5, color="#1DA1F2").encode(y=alt.Y("Tweets:Q"), tooltip=["Fecha", "Tweets"])
    line_mean = base.mark_line(color="black", strokeDash=[5, 5]).encode(y="Media:Q", tooltip=[alt.Tooltip("Media", format=".1f", title="Media 7d")])
    band = base.mark_area(opacity=0.2, color="gray").encode(y="Lower:Q", y2="Upper:Q")
    outliers = base.mark_circle(color="red", size=60).encode(y="Tweets:Q", tooltip=["Fecha", "Tweets"]).transform_filter(alt.datum.Tweets > alt.datum.Upper)

    chart = (band + bars + line_mean + outliers).properties(
        height=400,
        title="Volumen vs Media Móvil 7d (Puntos Rojos = Outliers > 1 Std)",
    )
    st.altair_chart(chart, use_container_width=True)

    st.divider()
    st.subheader("📆 Análisis de Trayectoria Semanal (Crucial para Polymarket)")

    col_A, col_B = st.columns([2, 1])

    with col_A:
        df_cum = daily_data.copy()
        df_cum["weekday"] = df_cum.index.dayofweek
        
        # The market week starts on Friday at 12:00 PM ET. Day 4.
        market_week_start_day = 4 

        # Find the most recent Friday
        today_et = df_cum.index.max()
        days_since_friday = (today_et.weekday() - market_week_start_day + 7) % 7
        current_week_start_date = today_et - pd.Timedelta(days=days_since_friday)

        current_week_data = df_cum[df_cum.index >= current_week_start_date].copy()
        current_week_data["cumsum"] = current_week_data["n_tweets"].cumsum()
        current_week_data["Type"] = "Semana Actual"

        last_week_start_date = current_week_start_date - pd.Timedelta(days=7)
        last_week_data = df_cum[
            (df_cum.index >= last_week_start_date) & (df_cum.index < current_week_start_date)
        ].copy()
        # Align weekday for comparison
        last_week_data["weekday"] = (last_week_data["weekday"] - market_week_start_day + 7) % 7
        last_week_data = last_week_data.sort_values("weekday")
        last_week_data["cumsum"] = last_week_data["n_tweets"].cumsum()
        last_week_data["Type"] = "Semana Anterior"

        three_months_ago = df_cum.index.max() - pd.DateOffset(months=3)
        hist_data = df_cum[df_cum.index >= three_months_ago].copy()
        
        # Adjust weekday for historical average
        hist_data["weekday_market"] = (hist_data.index.dayofweek - market_week_start_day + 7) % 7
        avg_week = (
            hist_data.groupby("weekday_market")["n_tweets"].mean().cumsum().reset_index()
        )
        avg_week.columns = ["weekday", "cumsum"]
        avg_week["Type"] = "Promedio (3 meses)"

        days_map = {0: "Vie", 1: "Sáb", 2: "Dom", 3: "Lun", 4: "Mar", 5: "Mié", 6: "Jue"}
        
        current_week_data["weekday"] = (current_week_data["weekday"] - market_week_start_day + 7) % 7

        combined_chart_data = pd.concat(
            [
                current_week_data[["weekday", "cumsum", "Type"]],
                last_week_data[["weekday", "cumsum", "Type"]],
                avg_week[["weekday", "cumsum", "Type"]],
            ],
        )
        combined_chart_data["Día"] = combined_chart_data["weekday"].map(days_map)

        chart_cum = (
            alt.Chart(combined_chart_data)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "weekday",
                    title="Día de la Semana del Mercado (Inicia Viernes)",
                    axis=alt.Axis(
                        values=list(days_map.keys()),
                        labelExpr="datum.label"
                    ),
                    sort=list(days_map.keys())
                ),
                y=alt.Y("cumsum", title="Tweets Acumulados"),
                color=alt.Color(
                    "Type",
                    scale=alt.Scale(
                        domain=["Semana Actual", "Semana Anterior", "Promedio (3 meses)"],
                        range=["red", "gray", "blue"],
                    ),
                ),
                tooltip=["Type", "Día", alt.Tooltip("cumsum", format=".0f")],
            )
            .properties(title="Carrera Semanal: Acumulado vs Histórico", height=350)
            .configure_axis_x(labelAngle=0)
        )
        # Replace labelExpr with actual labels for compatibility
        chart_cum.encoding.x.axis.labelExpr = "['Vie', 'Sáb', 'Dom', 'Lun', 'Mar', 'Mié', 'Jue'][datum.value]"


        st.altair_chart(chart_cum, use_container_width=True)

    with col_B:
        st.markdown("**Patrón Diario (Últimos 90 días)**")

        hist_data['weekday_label'] = hist_data.index.day_name()
        seasonality = hist_data.groupby("weekday_label")["n_tweets"].mean().reset_index()
        
        day_order = ["Friday", "Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"]
        seasonality['weekday_label'] = pd.Categorical(seasonality['weekday_label'], categories=day_order, ordered=True)
        seasonality = seasonality.sort_values('weekday_label')


        chart_bar = (
            alt.Chart(seasonality)
            .mark_bar()
            .encode(
                x=alt.X("n_tweets", title="Media Tweets"),
                y=alt.Y("weekday_label", sort=day_order, title=""),
                color=alt.Color("n_tweets", legend=None),
                tooltip=["weekday_label", alt.Tooltip("n_tweets", format=".1f")],
            )
            .properties(height=350)
        )
        st.altair_chart(chart_bar, use_container_width=True)

    today_market_weekday = (today_et.weekday() - market_week_start_day + 7) % 7
    
    avg_total = avg_week["cumsum"].max()
    avg_until_today_rows = avg_week[avg_week["weekday"] == today_market_weekday]

    if not avg_until_today_rows.empty:
        avg_until_today = avg_until_today_rows["cumsum"].values[0]
        expected_remaining = avg_total - avg_until_today
    else:
        expected_remaining = 0

    current_total = current_week_data["cumsum"].max()
    projected_total = current_total + expected_remaining

    st.info(
        f"""
    🧠 **Proyección Ingenua (Naive Projection):**
    
    Llevamos **{current_total}** tweets esta semana de mercado. Históricamente, en los días restantes Elon hace **{expected_remaining:.0f}** tweets más.
    
    👉 **Proyección Final Estimada: ~{projected_total:.0f} Tweets** (Si se comporta como el promedio reciente).
    """
    )

except Exception as e:
    st.error(f"Ocurrió un error al cargar o procesar los datos: {e}", icon="🔥")
    st.exception(e)