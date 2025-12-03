"""
main.py

This script is the main entry point for the Elon Musk Tweet Prediction Dashboard.
It uses Streamlit to provide an interactive user interface, orchestrating data loading,
metric calculation, model predictions, and visualization of trading opportunities.

The script delegates primary responsibilities to specialized modules within 'src/dashboard'
to maintain a modular and clean structure:
- `DashboardDataLoader`: Loads all data necessary, trained models, and risk parameters.
- `DashboardLogicProcessor`: Performs KPI calculations, hybrid predictions, and determines trading opportunities.
- `DashboardChartGenerator`: Generates Altair chart objects for visualization.

Main workflow:
1. Initial Streamlit setup and import handling.
2. Initialization of data, logic, and chart processors.
3. Loading of tweet data, Prophet models, and optimized risk parameters.
4. Displays a summary of the loaded model and the analyzed market.
5. Presents a statistical analysis of tweet activity with KPIs and charts.
6. Calculates and displays hybrid predictions and key metrics.
7. Calculates trading opportunities and displays the resulting table.
8. Visualizes the model's probability distribution against market prices.

This script is designed to run as a Streamlit application: `streamlit run main.py`.
"""

import asyncio
import os
import sys

import pandas as pd
import streamlit as st

# FIX for asyncio conflict between Streamlit and Playwright en Windows
if sys.platform == "win32" and sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# --- Internal Imports ---
try:
    # Use os.getcwd() for robustness, especially with Streamlit
    project_root = os.getcwd()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.dashboard.dashboard_chart_generator import DashboardChartGenerator
    from src.dashboard.dashboard_data_loader import DashboardDataLoader
    from src.dashboard.dashboard_logic_processor import DashboardLogicProcessor
except (ImportError, ModuleNotFoundError) as e:
    st.error(f"Import error. Ensure the folder structure is correct. Error: {e}")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="Elon Quant Dashboard", layout="wide", initial_sidebar_state="collapsed",
)

# --- MAIN UI ---
st.title("Elon Musk: Prediction and Quantitative Analysis Pipeline")

try:
    with st.spinner("Loading dashboard components and data... Please wait."):
        # Initialize Processors
        data_loader = DashboardDataLoader()
        logic_processor = DashboardLogicProcessor()
        chart_generator = DashboardChartGenerator()

        # Load all necessary data
        # Temporarily disable caching to debug stale data issue
        # @st.cache_data for efficiency
        granular_data, daily_data = data_loader.load_and_prepare_tweets_data()
        model_data = data_loader.load_prophet_model()
        risk_params = data_loader.load_risk_parameters()
        market_info = data_loader.fetch_market_data()

    prophet_model = model_data["model"]
    model_name = model_data.get("model_name", "Unknown")
    optimal_alpha = risk_params["alpha"]
    optimal_kelly = risk_params["kelly"]

    st.success(
        f"Model '{model_name}' loaded. Using Alpha: **{optimal_alpha:.4f}** and Kelly Frac: **{optimal_kelly:.2f}** (financially optimized).",
    )
    st.info(f"**Analyzed Market:** {market_info['market_question']}")

    # --- User Controls in Sidebar ---
    st.sidebar.title("View Options")

    historical_performance_df = st.cache_data(ttl=3600)(
        data_loader.load_historical_performance,
    )()

    # Prepare week options for selectbox
    week_options = ["Current Week"]
    if not historical_performance_df.empty:
        # Format week_start_date for display: YYYY-MM-DD
        # Ensure the index is localized before formatting, as st.cache_data returns localized.
        historical_weeks = historical_performance_df.index.strftime("%Y-%m-%d").tolist()
        week_options.extend(historical_weeks)

    selected_week = st.sidebar.selectbox("Select Week", week_options)

    if selected_week == "Current Week":
        # --- Current Week / Live Dashboard View ---
        st.sidebar.title("Trading Settings")
        bankroll = st.sidebar.number_input(
            "Enter your capital (Bankroll $):",
            min_value=100.0,
            value=1000.0,
            step=100.0,
        )

        # --- Statistical Analysis Section ---
        st.subheader("📈 Statistical Activity Analysis")
        st.markdown(
            "Analysis of historical tweet activity to contextualize Elon Musk's behavior.",
        )

        # Calculate KPIs
        kpis = logic_processor.calculate_kpis(daily_data)

        # Mostrar KPIs
        st.markdown("##### Key Recent Activity Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            label=f"Daily Mean ({kpis['current_month_str']})",
            value=f"{kpis['monthly_mean']:.1f}",
        )
        c2.metric(
            label="Yesterday vs Mean Deviation",
            value=f"{kpis['yesterday_val']:.0f}",
            delta=f"{kpis['deviation']:+.1f}",
        )
        c3.metric(
            label="Outlier Days (7d)",
            value=f"{kpis['outlier_days']} days",
            delta="High Volatility" if kpis["outlier_days"] >= 2 else "Stable",
        )
        c4.metric(label="Standard Deviation (7d)", value=f"{kpis['std_7d']:.2f}")

        # Statistical charts
        st.markdown("##### Activity Patterns (Last 6 Months)")
        col1, col2 = st.columns(2)
        chart_dow, chart_dist = chart_generator.generate_statistical_charts(
            daily_data, kpis["data_last_6_months"],
        )
        with col1:
            st.altair_chart(chart_dow, use_container_width=True)
        with col2:
            st.altair_chart(chart_dist, use_container_width=True)
        st.caption(
            "The charts show the average tweets per day of the week and the distribution of the total weekly tweet count.",
        )

        # --- Prediction and Market Section ---
        st.divider()
        st.subheader("🤖 Market Prediction and Edge Calculation")
        st.markdown(
            "Combines model prediction with real-time market data to identify trading opportunities.",
        )

        with st.spinner("Calculating opportunities..."):
            df_opportunities, pred_metrics = (
                logic_processor.calculate_trading_opportunities(
                    prophet_model=prophet_model,
                    optimal_alpha=optimal_alpha,
                    optimal_kelly=optimal_kelly,
                    market_info=market_info,
                    granular_data=granular_data,
                    bankroll=bankroll,
                )
            )

        # Mostrar Métricas de Predicción Híbrida
        st.markdown("##### Hybrid Prediction Breakdown")
        p1, p2, p3 = st.columns(3)
        p1.metric(
            "Total Hybrid Prediction", f"{pred_metrics['weekly_total_prediction']:.2f}",
        )
        p2.metric("Actual Tweets Counted", f"{pred_metrics['sum_of_actuals']}")
        p3.metric(
            "Predicted Tweets (future)",
            f"{pred_metrics['sum_of_predictions']:.2f}",
            delta=f"{pred_metrics['remaining_days_fraction']:.2f} remaining days",
        )
        st.caption(
            "Hybrid prediction combines tweets already published this week with the model's prediction for the remaining days.",
        )

        st.markdown("##### Trading Opportunities Table")
        st.dataframe(
            logic_processor.style_opportunities_df(df_opportunities),
            use_container_width=True,
        )
        st.caption(
            "The table compares the probability calculated by the model with the market price. 'Edge' is the difference between the two. 'Bet Size' is the recommended investment according to the fractional Kelly Criterion.",
        )

        # 5. Distribution Visualization
        st.divider()
        st.subheader("📊 Probability Distribution Visualization")
        st.markdown(
            "Visual comparison of the model's probability distribution (line) against market-implied probabilities (bars).",
        )

        final_chart = chart_generator.generate_probability_comparison_chart(
            df_opportunities,
        )
        st.altair_chart(final_chart, use_container_width=True)
        st.caption(
            "Areas where the line (model) is above the bars (market) represent a positive 'edge', suggesting a 'buy' bet.",
        )

    else:
        # --- Historical Performance View ---
        st.subheader(f"🕰️ Historical Performance for Week starting {selected_week}")

        # Filter for the selected week
        # selected_week_dt = pd.to_datetime(selected_week, tz='UTC') # Old line
        selected_week_dt = pd.to_datetime(selected_week).tz_localize(
            "UTC",
        )  # Fix for TypeError
        week_data = historical_performance_df.loc[selected_week_dt]

        if not week_data.empty:
            col_hp1, col_hp2 = st.columns(2)
            with col_hp1:
                st.metric(
                    label="Model Prediction (y_pred)",
                    value=f"{week_data['y_pred']:.2f}",
                )
            with col_hp2:
                st.metric(
                    label="Actual Tweets (y_true)", value=f"{week_data['y_true']:.0f}",
                )

            st.divider()

            # --- Chart for Historical Week ---
            # Filter daily_data for the selected week
            # The `daily_data` index is datetime. The `selected_week_dt` is the start of the week.
            # Assuming a week is 7 days from `selected_week_dt`
            daily_data_for_selected_week = daily_data[
                (daily_data.index >= selected_week_dt)
                & (daily_data.index < selected_week_dt + pd.Timedelta(days=7))
            ]

            if not daily_data_for_selected_week.empty:
                historical_chart = chart_generator.generate_historical_week_chart(
                    daily_data_for_selected_week,
                )
                st.altair_chart(historical_chart, use_container_width=True)
            else:
                st.info(
                    "No daily tweet activity data available for this historical week.",
                )

        else:
            st.warning("No historical performance data available for this week.")

except Exception as e:
    st.error(f"An error occurred in the main pipeline: {e}")
    st.exception(e)
