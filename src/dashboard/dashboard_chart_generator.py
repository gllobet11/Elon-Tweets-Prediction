"""
dashboard_chart_generator.py

Este módulo se encarga exclusivamente de la generación de objetos de gráficos
utilizando la librería Altair. Su propósito es centralizar toda la lógica
de visualización, manteniendo la interfaz de usuario de Streamlit limpia
y enfocada en la presentación.

Proporciona funciones para generar gráficos estadísticos de la actividad
de tweets y gráficos de comparación de probabilidades del modelo frente al mercado.
"""

import pandas as pd
import altair as alt


class DashboardChartGenerator:
    """
    Clase para generar los objetos de gráficos de Altair para el dashboard.
    """
    def __init__(self):
        """
        Inicializa el generador de gráficos.
        """
        pass

    def generate_statistical_charts(self, daily_data: pd.Series, data_last_6_months: pd.DataFrame) -> tuple[alt.Chart, alt.Chart]:
        """
        Genera gráficos de Altair para el análisis estadístico de la actividad de tweets.

        Args:
            daily_data (pd.Series): Serie de conteos diarios de tweets (para contexto).
            data_last_6_months (pd.DataFrame): DataFrame con datos de tweets de los últimos 6 meses.

        Returns:
            tuple[alt.Chart, alt.Chart]: Una tupla que contiene los gráficos de Altair:
                - chart_dow (alt.Chart): Gráfico de barras de la media de tweets por día de la semana.
                - chart_dist (alt.Chart): Gráfico de barras de la distribución de tweets semanales.
        """
        data_last_6_months['day_of_week'] = data_last_6_months.index.day_name()
        avg_by_day = data_last_6_months.groupby('day_of_week')['n_tweets'].mean().reset_index()
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        avg_by_day['day_of_week'] = pd.Categorical(avg_by_day['day_of_week'], categories=day_order, ordered=True)
        
        chart_dow = alt.Chart(avg_by_day.sort_values('day_of_week')).mark_bar().encode(
            x=alt.X('day_of_week', sort=None, title="Día de la Semana"),
            y=alt.Y('n_tweets', title="Media de Tweets")
        ).properties(title="Media de Tweets por Día (Últimos 6 meses)", height=300)
        
        weekly_sums = data_last_6_months.resample('W-MON')['n_tweets'].sum().reset_index()
        chart_dist = alt.Chart(weekly_sums).mark_bar().encode(
            alt.X("n_tweets", bin=alt.Bin(maxbins=20), title="Total de Tweets Semanales"),
            alt.Y('count()', title="Frecuencia (Nº de Semanas)")
        ).properties(title="Distribución de Tweets Semanales (Últimos 6 meses)", height=300)
        
        return chart_dow, chart_dist

    def generate_probability_comparison_chart(self, df_opportunities: pd.DataFrame) -> alt.Chart:
        """
        Genera un gráfico de Altair en capas que compara las probabilidades del modelo
        con los precios de mercado para cada bin.

        Args:
            df_opportunities (pd.DataFrame): DataFrame con las oportunidades de trading,
                                            incluyendo precios de mercado y probabilidades del modelo por bin.

        Returns:
            alt.Chart: Objeto de gráfico de Altair con la comparación de probabilidades.
        """
        # 1. Añadir puntos medios numéricos a los datos para el eje X
        def get_midpoint(bin_label: str) -> float:
            """Calcula el punto medio de una etiqueta de bin."""
            if '+' in bin_label:
                return int(bin_label.replace('+', '').replace(',', '')) + 20 # Offset para bins abiertos
            parts = bin_label.replace(',', '').split('-')
            return (int(parts[0]) + int(parts[1])) / 2
        
        df_opportunities['Midpoint'] = df_opportunities['Bin'].apply(get_midpoint)

        # 2. Crear las capas del gráfico
        # Capa base para compartir ejes
        base = alt.Chart(df_opportunities).encode(
            x=alt.X('Midpoint:Q', title='Número de Tweets', axis=alt.Axis(labelAngle=-45))
        )

        # Capa 1: Precios de Mercado (Barras)
        market_bars = base.mark_bar(
            width=20,  # Ancho fijo para las barras
            opacity=0.7,
            color='#ff7f0e' # Naranja para el mercado
        ).encode(
            y=alt.Y('Mkt Price:Q', title='Probabilidad', axis=alt.Axis(format='%')),
            tooltip=[
                alt.Tooltip('Bin', title='Bin'),
                alt.Tooltip('Mkt Price', title='Mkt. Price', format='.3%')
            ]
        ).properties(
            width=600  # Ancho ajustado
        )
        
        # Capa 2: Probabilidad del Modelo (Línea + Puntos)
        model_line = base.mark_line(
            color='#1f77b4' # Azul para el modelo
        ).encode(
            y=alt.Y('My Model:Q', title='Probabilidad')
        )
        
        model_points = base.mark_point(
            color='#1f77b4',
            filled=True,
            size=60
        ).encode(
            y=alt.Y('My Model:Q'),
            tooltip=[
                alt.Tooltip('Bin', title='Bin'),
                alt.Tooltip('My Model', title='Model Prob.', format='.3%')
            ]
        )

        # 3. Combinar y renderizar el gráfico
        final_chart = (market_bars + model_line + model_points).properties(
            title='Comparación de Probabilidades: Modelo (Línea) vs. Mercado (Barras)'
        ).interactive()

        return final_chart