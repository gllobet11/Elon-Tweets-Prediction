"""
dashboard_chart_generator.py
Genera gráficos Altair para el dashboard.
"""
import altair as alt
import pandas as pd

class DashboardChartGenerator:
    def __init__(self):
        pass

    def generate_statistical_charts(self, daily_data: pd.DataFrame, data_last_6_months: pd.DataFrame):
        """Genera gráficos de actividad semanal y distribución histórica."""
        if daily_data.empty:
            return alt.Chart(pd.DataFrame()).mark_text(text="No data"), alt.Chart(pd.DataFrame()).mark_text(text="No data")

        # 1. Gráfico por Día de la Semana
        daily_data['day_name'] = daily_data.index.day_name()
        # Ordenar días correctamente
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        chart_dow = (
            alt.Chart(daily_data.reset_index())
            .mark_bar()
            .encode(
                x=alt.X("day_name", sort=days_order, title="Day of Week"),
                y=alt.Y("mean(n_tweets)", title="Avg Tweets"),
                color=alt.Color("day_name", legend=None),
                tooltip=["day_name", "mean(n_tweets)"]
            )
            .properties(title="Activity by Day of Week (All Time)")
        )

        # 2. Histograma de Distribución (Últimos 6 meses)
        chart_dist = (
            alt.Chart(data_last_6_months.reset_index())
            .mark_bar()
            .encode(
                x=alt.X("n_tweets", bin=alt.Bin(maxbins=30), title="Daily Tweet Count"),
                y=alt.Y("count()", title="Frequency"),
                color=alt.value("orange")
            )
            .properties(title="Activity Distribution (Last 6 Months)")
        )

        return chart_dow, chart_dist

    def generate_probability_comparison_chart(self, df_opportunities: pd.DataFrame):
        """
        Compara la distribución del modelo (Línea) vs Mercado (Barras).
        """
        if df_opportunities.empty:
            return alt.Chart(pd.DataFrame()).mark_text(text="No Opportunities Data")

        # Preparar datos
        # Necesitamos un eje X numérico para Altair si queremos línea continua, 
        # pero 'Bin' es categórico (ej: '100-119'). Usaremos categórico para simplificar.
        
        base = alt.Chart(df_opportunities).encode(x=alt.X("Bin", sort=None))

        # Barras = Mercado
        bars = base.mark_bar(opacity=0.3, color="gray").encode(
            y=alt.Y("Mkt Price", title="Probability"),
            tooltip=["Bin", "Mkt Price"]
        )

        # Línea = Modelo
        # Usamos mark_line + mark_point para ver la tendencia del modelo
        line = base.mark_line(color="green", strokeWidth=3).encode(
            y="My Model",
            tooltip=["Bin", "My Model"]
        )
        points = base.mark_circle(color="green").encode(
            y="My Model"
        )

        chart = (bars + line + points).properties(
            title="Model (Green) vs Market (Gray) Probability Distribution"
        ).interactive()

        return chart

    def generate_historical_week_chart(self, daily_data_week: pd.DataFrame):
        """Gráfico de la evolución de tweets en una semana histórica específica."""
        if daily_data_week.empty:
            return alt.Chart(pd.DataFrame()).mark_text(text="No Data")

        chart = (
            alt.Chart(daily_data_week.reset_index())
            .mark_bar()
            .encode(
                x=alt.X("created_at", title="Date", axis=alt.Axis(format="%Y-%m-%d")),
                y=alt.Y("n_tweets", title="Tweets"),
                tooltip=["created_at", "n_tweets"]
            )
            .properties(title="Daily Activity for Selected Week")
        )
        return chart