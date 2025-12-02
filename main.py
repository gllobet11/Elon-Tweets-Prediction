"""
main.py

Este script es el punto de entrada principal para el Dashboard de Predicción de Tweets de Elon Musk.
Utiliza Streamlit para proporcionar una interfaz de usuario interactiva, orquestando la carga de datos,
el cálculo de métricas, las predicciones del modelo y la visualización de oportunidades de trading.

El script delega las responsabilidades principales a módulos especializados dentro de 'src/dashboard'
para mantener una estructura modular y limpia:
- `DashboardDataLoader`: Carga todos los datos necesarios, modelos entrenados y parámetros de riesgo.
- `DashboardLogicProcessor`: Realiza los cálculos de KPIs, predicciones híbridas y determina las oportunidades de trading.
- `DashboardChartGenerator`: Genera los objetos de gráficos de Altair para su visualización.

Workflow principal:
1. Configuración inicial de Streamlit y manejo de importaciones.
2. Inicialización de los procesadores de datos, lógica y gráficos.
3. Carga de datos de tweets, modelos Prophet y parámetros de riesgo optimizados.
4. Muestra un resumen del modelo cargado y el mercado analizado.
5. Presenta un análisis estadístico de la actividad de tweets con KPIs y gráficos.
6. Calcula y muestra las predicciones híbridas y las métricas clave.
7. Calcula las oportunidades de trading y muestra la tabla resultante.
8. Visualiza la distribución de probabilidades del modelo frente a los precios de mercado.

Este script está diseñado para ejecutarse como una aplicación de Streamlit: `streamlit run main.py`.
"""

import asyncio
import sys

# FIX para el conflicto de asyncio entre Streamlit y Playwright en Windows
if sys.platform == "win32" and sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timedelta
import os


# --- Internal Imports ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    

    from src.dashboard.dashboard_data_loader import DashboardDataLoader
    from src.dashboard.dashboard_logic_processor import DashboardLogicProcessor
    from src.dashboard.dashboard_chart_generator import DashboardChartGenerator
except (ImportError, ModuleNotFoundError) as e:
    st.error(f"Error de importación. Asegúrate de que la estructura de carpetas es correcta. Error: {e}")
    st.stop()

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Elon Quant Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- UI PRINCIPAL ---
st.title("Elon Musk: Pipeline de Predicción y Análisis Cuantitativo")

try:
    # Initialize Processors
    data_loader = DashboardDataLoader()
    logic_processor = DashboardLogicProcessor()
    chart_generator = DashboardChartGenerator()

    # Load all necessary data
    # @st.cache_data for efficiency
    granular_data, daily_data = st.cache_data(ttl=3600)(data_loader.load_and_prepare_tweets_data)() # Re-add Streamlit caching here
    model_data = st.cache_data(ttl=3600)(data_loader.load_prophet_model)()
    risk_params = st.cache_data(ttl=3600)(data_loader.load_risk_parameters)()
    market_info = st.cache_data(ttl=3600)(data_loader.fetch_market_data)()

    prophet_model = model_data['model']
    model_name = model_data.get('model_name', 'Unknown')
    optimal_alpha = risk_params['alpha']
    optimal_kelly = risk_params['kelly']

    st.success(f"Modelo '{model_name}' cargado. Usando Alpha: **{optimal_alpha:.4f}** y Kelly Frac: **{optimal_kelly:.2f}** (optimizados financieramente).")
    st.info(f"**Mercado Analizado:** {market_info['market_question']}")

        # --- Sección de Análisis Estadístico ---
        st.subheader("📈 Análisis Estadístico de Actividad")
        st.markdown("Análisis de la actividad histórica de tweets para contextualizar el comportamiento de Elon Musk.")
    
        # Calculate KPIs
        kpis = logic_processor.calculate_kpis(daily_data)
    
        # Mostrar KPIs
        st.markdown("##### Métricas Clave de Actividad Reciente")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(label=f"Media Diaria ({kpis['current_month_str']})", value=f"{kpis['monthly_mean']:.1f}")
        c2.metric(label="Desviación Ayer vs Media", value=f"{kpis['yesterday_val']:.0f}", delta=f"{kpis['deviation']:+.1f}")
        c3.metric(label="Días 'Outlier' (7d)", value=f"{kpis['outlier_days']} días", delta="Alta Volatilidad" if kpis['outlier_days'] >= 2 else "Estable")
        c4.metric(label="Desviación Típica (7d)", value=f"{kpis['std_7d']:.2f}")
    
        # Gráficos estadísticos
        st.markdown("##### Patrones de Actividad (Últimos 6 Meses)")
        col1, col2 = st.columns(2)
        chart_dow, chart_dist = chart_generator.generate_statistical_charts(daily_data, kpis['data_last_6_months'])
        with col1:
            st.altair_chart(chart_dow, use_container_width=True)
        with col2:
            st.altair_chart(chart_dist, use_container_width=True)
        st.caption("Los gráficos muestran la media de tweets por día de la semana y la distribución de la cantidad total de tweets semanales.")
    
        # --- Controles de Usuario en la Barra Lateral ---
        st.sidebar.title("Configuración de Trading")
        bankroll = st.sidebar.number_input("Introduce tu capital (Bankroll $):", min_value=100.0, value=1000.0, step=100.0)
    
        # --- Sección de Predicción y Mercado ---
        st.divider()
        st.subheader("🤖 Predicción de Mercado y Cálculo de Edge")
        st.markdown("Combinación de la predicción del modelo con los datos del mercado en tiempo real para identificar oportunidades de trading.")
        
        with st.spinner('Calculando oportunidades...'):
            df_opportunities, pred_metrics = logic_processor.calculate_trading_opportunities(
                prophet_model=prophet_model,
                optimal_alpha=optimal_alpha,
                optimal_kelly=optimal_kelly,
                market_info=market_info,
                granular_data=granular_data,
                bankroll=bankroll
            )
    
        # Mostrar Métricas de Predicción Híbrida
        st.markdown("##### Desglose de la Predicción Híbrida")
        p1, p2, p3 = st.columns(3)
        p1.metric("Predicción Híbrida Total", f"{pred_metrics['weekly_total_prediction']:.2f}")
        p2.metric("Tweets Reales Contados", f"{pred_metrics['sum_of_actuals']}")
        p3.metric("Tweets Predichos (futuro)", f"{pred_metrics['sum_of_predictions']:.2f}", delta=f"{pred_metrics['remaining_days_fraction']:.2f} días restantes")
        st.caption("La predicción híbrida combina los tweets ya publicados esta semana con la predicción del modelo para los días restantes.")
    
        st.markdown("##### Tabla de Oportunidades de Trading")
        st.dataframe(logic_processor.style_opportunities_df(df_opportunities), use_container_width=True)
        st.caption("La tabla compara la probabilidad calculada por el modelo con el precio del mercado. El 'Edge' es la diferencia entre ambos. 'Bet Size' es la inversión recomendada según el Criterio de Kelly fraccional.")
    
        # 5. Visualización de la Distribución
        st.divider()
        st.subheader("📊 Visualización de la Distribución de Probabilidades")
        st.markdown("Comparación visual de la distribución de probabilidad del modelo (línea) frente a las probabilidades implícitas en los precios del mercado (barras).")
    
        final_chart = chart_generator.generate_probability_comparison_chart(df_opportunities)
        st.altair_chart(final_chart, use_container_width=True)
        st.caption("Las áreas donde la línea (modelo) está por encima de las barras (mercado) representan un 'edge' positivo, sugiriendo una apuesta de 'compra'.")
except Exception as e:
    st.error(f"Ocurrió un error en el pipeline principal: {e}")
    st.exception(e)